# cage_cropper.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
import cv2
import numpy as np

# ---------- data model ----------
@dataclass
class CropPlan:
    video_path: str
    bg_size: Tuple[int, int]          # (H, W) in original
    pre_crop_roi: Tuple[int, int, int, int]   # (x0, y0, w, h) in original
    rim_rect_trim: Tuple[Tuple[float, float], Tuple[float, float], float]  # ((cx,cy),(w,h),ang) in pre-crop coords
    rim_density: float
    fit_score: float
    H_pre: np.ndarray                 # 3x3: original -> pre-crop
    H_rect: np.ndarray                # 3x3: pre-crop -> rectified crop
    H_post: np.ndarray                # 3x3: optional post-rotate (90°)
    H_total: np.ndarray               # 3x3: original -> final crop
    out_size: Tuple[int, int]         # (H, W) of final crop
    rotated_90: bool                  # True if long edge needed 90° CW
    rotation_rad: float               # rotation applied to final crop (radians)

# ---------- utilities ----------
def _order_box_pts_ccw(box4: np.ndarray) -> np.ndarray:
    s = box4.sum(axis=1)
    d = np.diff(box4, axis=1)[:, 0]
    tl = box4[np.argmin(s)]
    br = box4[np.argmax(s)]
    tr = box4[np.argmin(d)]
    bl = box4[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _rim_edges_1px(binary_u8: np.ndarray) -> np.uint8:
    H, W = binary_u8.shape[:2]
    def _odd(n): return n if n % 2 == 1 else n + 1
    def _ellip(frac, lo=3, hi=11):
        k = int(round(min(H, W) * frac)); k = max(lo, min(hi, k)); k = _odd(k)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, _ellip(0.02), iterations=1)
    grad   = cv2.morphologyEx(opened,    cv2.MORPH_GRADIENT, _ellip(0.004), iterations=1)
    grad   = cv2.dilate(grad,            _ellip(0.003), iterations=1)
    return (grad > 0).astype(np.uint8)

def _rim_fit_score_perimeter(edges01: np.uint8, rect, tol_px: int = 6) -> float:
    H, W = edges01.shape[:2]
    box = cv2.boxPoints(rect).astype(np.int32)
    perim = np.zeros((H, W), np.uint8)
    cv2.polylines(perim, [box], True, 255, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol_px+1, 2*tol_px+1))
    rim_dil = cv2.dilate(edges01.astype(np.uint8), k, 1) * 255
    agree = cv2.bitwise_and(perim, rim_dil)
    n_perim = int(perim.sum() // 255)
    return 0.0 if n_perim == 0 else float(agree.sum() // 255) / n_perim

def _median_background(video_path: Path, sample_step: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.arange(0, n, max(1, sample_step), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read for background")
    bg = np.median(np.stack(frames, 0), axis=0).astype(np.uint8)
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) if bg.ndim == 3 else bg

# ---------- coarse pre-crop (largest rectangle / bbox) ----------
def _precrop_roi_from_bg(bg_gray: np.ndarray,
                         thresh: int = 60,
                         dilate_ks: int = 13,
                         erode_ks: int = 7,
                         expand_pct: float = 15.0,
                         save_debug: Optional[Path] = None) -> Tuple[int, int, int, int]:
    """Return (x0,y0,w,h) of coarse ROI in original coordinates."""
    _, bin_u8 = cv2.threshold(bg_gray, thresh, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(save_debug / "thresh_debug.png"), bin_u8) if save_debug is not None else None
    edges = cv2.Canny(bin_u8, 50, 150, apertureSize=3, L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ks, dilate_ks)), 1)
    edges = cv2.erode(edges,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ks, erode_ks)), 1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        H, W = bg_gray.shape[:2];  return 0, 0, W, H
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)  # robust even if we don't get a perfect 4-pt approx
    # expand
    cx, cy = x + w/2.0, y + h/2.0
    w2 = int(round(w * (1.0 + expand_pct/100.0)))
    h2 = int(round(h * (1.0 + expand_pct/100.0)))
    x0 = int(round(cx - w2/2.0)); y0 = int(round(cy - h2/2.0))
    H, W = bg_gray.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x0 + w2); y1 = min(H, y0 + h2)
    return x0, y0, x1 - x0, y1 - y0

# ---------- main API ----------
def make_plan(video_path: Path,
              sample_step: int = 500,
              pad_px: int = 2,
              save_debug: Optional[Path] = None,
              canonical_long_edge: bool = True) -> Tuple[np.ndarray, CropPlan]:
    """
    Returns (cropped_background_u8, CropPlan).
    H_total maps ORIGINAL frames directly into the final crop (use with cv2.warpPerspective).
    """
    # 1) Background and coarse pre-crop
    thresh = 90
    bg_gray = _median_background(video_path, sample_step)
    H0, W0 = bg_gray.shape[:2]
    x0, y0, w0, h0 = _precrop_roi_from_bg(bg_gray, thresh=thresh, save_debug=save_debug)
    trim = bg_gray[y0:y0+h0, x0:x0+w0]

    # translation matrix original -> pre-crop
    H_pre = np.array([[1., 0., -x0],
                      [0., 1., -y0],
                      [0., 0.,  1. ]], dtype=np.float32)

    # 2) Rim edges and min-area rectangle (in pre-crop coords)
    _, bin_u8 = cv2.threshold(trim, thresh, 255, cv2.THRESH_BINARY)
    # show threshold debug
    cv2.imwrite(str(save_debug / f"{Path(video_path).stem}_precrop_thresh.png"), bin_u8) if save_debug is not None else None
    edges01 = _rim_edges_1px(bin_u8)
    rim_density = float(edges01.mean())
    
    # 1. Close the edges to bridge gaps in the rim (makes it one solid loop)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed_edges = cv2.morphologyEx(edges01, cv2.MORPH_CLOSE, kernel)

    # 2. Use RETR_EXTERNAL to grab ONLY the outermost boundaries (ignores inner bedding/dust)
    cnts, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        raise RuntimeError("Rim too sparse; cannot find contours.")

    # 3. Filter using spatial heuristics: the cage MUST be large relative to the pre-crop
    H_trim, W_trim = trim.shape[:2]
    valid_cnts = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # The rim should span at least 60% of the pre-crop's width and height
        if w > 0.6 * W_trim and h > 0.6 * H_trim:
            valid_cnts.append(c)

    # Fallback to the largest contour by area if our strict heuristic misses
    if not valid_cnts:
        best_cnt = max(cnts, key=cv2.contourArea)
    else:
        # If multiple large contours exist, take the one with the largest bounding area
        best_cnt = max(valid_cnts, key=cv2.contourArea)

    if cv2.contourArea(best_cnt) < 100:
         raise RuntimeError("No significant cage rim found after filtering.")

    # 4. Calculate the minAreaRect on this single, perfectly isolated contour
    rect = cv2.minAreaRect(best_cnt.astype(np.float32))
    fit = _rim_fit_score_perimeter(edges01, rect, tol_px=6)

    # normalize so width >= height
    (cx, cy), (rw, rh), ang = rect
    if rh > rw:
        rw, rh = rh, rw
        ang += 90.0
    rect = ((cx, cy), (rw, rh), ang)

    # 3) Build canonical warp (pre-crop -> rectified crop)
    src = _order_box_pts_ccw(cv2.boxPoints(rect).astype(np.float32))
    out_W = int(round(rw)) + 2*pad_px
    out_H = int(round(rh)) + 2*pad_px
    dst = np.array([[pad_px,          pad_px],
                    [out_W-1-pad_px,  pad_px],
                    [out_W-1-pad_px,  out_H-1-pad_px],
                    [pad_px,          out_H-1-pad_px]], dtype=np.float32)
    H_rect = cv2.getPerspectiveTransform(src, dst)

    # 4) Optional 90° clockwise ensure long edge horizontal
    rotated_90 = False
    H_post = np.eye(3, dtype=np.float32)
    if canonical_long_edge and out_H > out_W:
        # (x,y) -> (y, W-1-x)
        H_post = np.array([[0., 1., 0.],
                           [-1., 0., out_W-1.],
                           [0., 0., 1.]], dtype=np.float32)
        out_W, out_H = out_H, out_W
        rotated_90 = True

    # 5) Compose total transform from ORIGINAL -> FINAL
    H_total = (H_post @ H_rect @ H_pre).astype(np.float32)

    # 6) Produce cropped background for preview
    bg_crop = cv2.warpPerspective(bg_gray, H_total, (out_W, out_H),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if save_debug is not None:
        save_debug.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_debug / f"{Path(video_path).stem}_bg_gray.png"), bg_gray)
        vis_edges = cv2.cvtColor(trim, cv2.COLOR_GRAY2BGR)
        vis_edges[edges01 > 0] = (0, 255, 0)
        cv2.imwrite(str(save_debug / f"{Path(video_path).stem}_precrop_trim.png"), trim)
        cv2.imwrite(str(save_debug / f"{Path(video_path).stem}_rim_overlay.png"), vis_edges)
        box = cv2.boxPoints(rect).astype(np.int32)
        vis_box = cv2.cvtColor(trim, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_box, [box], 0, (0, 0, 255), 2)
        cv2.imwrite(str(save_debug / f"{Path(video_path).stem}_rotrect.png"), vis_box)
        cv2.imwrite(str(save_debug / f"{Path(video_path).stem}_final_bg_crop.png"), bg_crop)

    plan = CropPlan(
        video_path=str(video_path),
        bg_size=(H0, W0),
        pre_crop_roi=(x0, y0, w0, h0),
        rim_rect_trim=((float(cx), float(cy)), (float(rw), float(rh)), float(ang)),
        rim_density=rim_density,
        fit_score=fit,
        H_pre=H_pre, H_rect=H_rect, H_post=H_post, H_total=H_total,
        out_size=(out_H, out_W),
        rotated_90=rotated_90,
        rotation_rad=float(np.arctan2(float(H_total[1, 0]), float(H_total[0, 0]))),
    )
    return bg_crop, plan

# ---------- apply to any frame ----------
def apply_plan_to_frame(frame_bgr_or_gray: np.ndarray, plan: CropPlan) -> np.ndarray:
    """Warp any frame from the same video to the canonical crop."""
    gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY) if frame_bgr_or_gray.ndim == 3 else frame_bgr_or_gray
    H, W = plan.out_size
    return cv2.warpPerspective(gray, plan.H_total, (plan.out_size[1], plan.out_size[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
