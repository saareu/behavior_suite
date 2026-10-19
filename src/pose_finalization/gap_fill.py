"""Configurable internal gap filling for dense 120 Hz pose trajectories.

Fills only bounded internal gaps. Leading/trailing gaps, protected samples,
and gaps longer than the Kalman threshold remain unresolved (NaN).

Default policy at 120 Hz:
  1–20 frames   → linear interpolation
  21–140 frames → constant-velocity Kalman + RTS
  >140 frames   → leave NaN
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_finalization.contracts import (
    DEFAULT_IMPUTATION_EXCLUDED_NODES,
    GAP_KALMAN_MAX_FRAMES_DEFAULT,
    GAP_LINEAR_MAX_FRAMES_DEFAULT,
    HEADSTAGE_NODE_NAME,
    PROVENANCE_KALMAN_GAP_FILL,
    PROVENANCE_LINEAR_GAP_FILL,
    PoseFinalizationError,
)


@dataclass(frozen=True, slots=True)
class GapFillConfig:
    """Configurable gap-fill thresholds and Kalman noise parameters."""

    linear_max_frames: int = GAP_LINEAR_MAX_FRAMES_DEFAULT
    kalman_max_frames: int = GAP_KALMAN_MAX_FRAMES_DEFAULT
    process_var: float = 10.0
    measurement_var: float = 1.0
    excluded_nodes: tuple[str, ...] = DEFAULT_IMPUTATION_EXCLUDED_NODES

    def __post_init__(self) -> None:
        if self.linear_max_frames < 0:
            raise PoseFinalizationError("linear_max_frames must be non-negative.")
        if self.kalman_max_frames < self.linear_max_frames:
            raise PoseFinalizationError(
                "kalman_max_frames must be >= linear_max_frames."
            )
        if not np.isfinite(self.process_var) or self.process_var <= 0.0:
            raise PoseFinalizationError("process_var must be finite and positive.")
        if not np.isfinite(self.measurement_var) or self.measurement_var <= 0.0:
            raise PoseFinalizationError("measurement_var must be finite and positive.")


def fill_coordinate_gaps(
    *,
    x: np.ndarray,
    y: np.ndarray,
    provenance: np.ndarray,
    protected: np.ndarray,
    node_names: np.ndarray,
    config: GapFillConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill bounded internal gaps in dense (sample, track, node) arrays.

    Preserves existing finite coordinates. Never writes into protected samples.
    Excluded nodes (default: headstage) are left unchanged.
    """

    cfg = config if config is not None else GapFillConfig()
    if x.shape != y.shape or x.shape != provenance.shape or x.shape != protected.shape:
        raise PoseFinalizationError("Gap-fill array shapes must match.")
    if x.ndim != 3:
        raise PoseFinalizationError("Gap-fill arrays must be (sample, track, node).")
    if node_names.shape != (x.shape[2],):
        raise PoseFinalizationError("node_names length must match the node axis.")

    out_x = np.array(x, dtype=np.float64, copy=True)
    out_y = np.array(y, dtype=np.float64, copy=True)
    out_prov = np.array(provenance, dtype=np.uint8, copy=True)
    excluded = {str(name) for name in cfg.excluded_nodes}

    for track_i in range(out_x.shape[1]):
        for node_i in range(out_x.shape[2]):
            node_name = str(node_names[node_i])
            if node_name in excluded or node_name == HEADSTAGE_NODE_NAME:
                continue
            _fill_series_xy(
                out_x[:, track_i, node_i],
                out_y[:, track_i, node_i],
                out_prov[:, track_i, node_i],
                protected[:, track_i, node_i],
                cfg,
            )
    return out_x, out_y, out_prov


def _fill_series_xy(
    x: np.ndarray,
    y: np.ndarray,
    provenance: np.ndarray,
    protected: np.ndarray,
    config: GapFillConfig,
) -> None:
    """In-place fill of one (track, node) trajectory."""

    n = x.shape[0]
    # Valid anchors: finite coordinates that are not manually protected.
    valid = np.isfinite(x) & np.isfinite(y) & ~protected
    # Fillable gaps: non-valid, non-protected samples only.
    missing = (~valid) & ~protected

    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < n and missing[i]:
            i += 1
        end = i  # exclusive
        gap_len = end - start
        left = start - 1
        right = end
        # Leading / trailing gaps remain unresolved.
        if left < 0 or right >= n:
            continue
        if not valid[left] or not valid[right]:
            continue
        # Never bridge a protected sample between the bounding observations.
        if np.any(protected[left + 1 : right]):
            continue

        if gap_len <= 0:
            continue
        if gap_len <= config.linear_max_frames:
            _linear_fill_segment(x, y, provenance, left, right, start, end)
        elif gap_len <= config.kalman_max_frames:
            _kalman_rts_fill_segment(
                x,
                y,
                provenance,
                left,
                right,
                start,
                end,
                process_var=config.process_var,
                measurement_var=config.measurement_var,
            )
        # else: leave NaN



def _linear_fill_segment(
    x: np.ndarray,
    y: np.ndarray,
    provenance: np.ndarray,
    left: int,
    right: int,
    start: int,
    end: int,
) -> None:
    span = float(right - left)
    for idx in range(start, end):
        w = (idx - left) / span
        x[idx] = (1.0 - w) * x[left] + w * x[right]
        y[idx] = (1.0 - w) * y[left] + w * y[right]
        provenance[idx] = PROVENANCE_LINEAR_GAP_FILL


def _kalman_rts_fill_segment(
    x: np.ndarray,
    y: np.ndarray,
    provenance: np.ndarray,
    left: int,
    right: int,
    start: int,
    end: int,
    *,
    process_var: float,
    measurement_var: float,
) -> None:
    """Fill ``[start, end)`` using CV Kalman + RTS with endpoints as observations."""

    filled_x = _cv_kalman_rts_1d(
        x[left : right + 1].copy(),
        process_var=process_var,
        measurement_var=measurement_var,
    )
    filled_y = _cv_kalman_rts_1d(
        y[left : right + 1].copy(),
        process_var=process_var,
        measurement_var=measurement_var,
    )
    # Map local indices: global left → local 0, global right → local -1.
    for idx in range(start, end):
        local = idx - left
        x[idx] = filled_x[local]
        y[idx] = filled_y[local]
        provenance[idx] = PROVENANCE_KALMAN_GAP_FILL


def _cv_kalman_rts_1d(
    series: np.ndarray,
    *,
    process_var: float,
    measurement_var: float,
) -> np.ndarray:
    """Constant-velocity Kalman filter + RTS smoother for one 1D series.

    Finite samples are treated as measurements. NaN samples are predicted only.
    Returns smoothed positions for every index; callers overwrite only gaps.
    """

    n = int(series.shape[0])
    if n == 0:
        return series

    # State: [position, velocity]
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    Q = process_var * np.array(
        [[1.0 / 4.0, 1.0 / 2.0], [1.0 / 2.0, 1.0]],
        dtype=np.float64,
    )
    R = np.array([[measurement_var]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)

    x_filt = np.zeros((n, 2), dtype=np.float64)
    P_filt = np.zeros((n, 2, 2), dtype=np.float64)
    x_pred = np.zeros((n, 2), dtype=np.float64)
    P_pred = np.zeros((n, 2, 2), dtype=np.float64)

    # Initialize from the first finite observation.
    first = int(np.flatnonzero(np.isfinite(series))[0])
    x_est = np.array([float(series[first]), 0.0], dtype=np.float64)
    P_est = np.eye(2, dtype=np.float64) * measurement_var

    for k in range(n):
        if k == 0:
            x_pred[k] = x_est
            P_pred[k] = P_est
        else:
            x_pred[k] = F @ x_filt[k - 1]
            P_pred[k] = F @ P_filt[k - 1] @ F.T + Q

        z = series[k]
        if np.isfinite(z):
            innov = float(z) - float((H @ x_pred[k])[0])
            S = H @ P_pred[k] @ H.T + R
            K = P_pred[k] @ H.T @ np.linalg.inv(S)
            x_filt[k] = x_pred[k] + (K[:, 0] * innov)
            P_filt[k] = (I - K @ H) @ P_pred[k]
        else:
            x_filt[k] = x_pred[k]
            P_filt[k] = P_pred[k]

    # RTS backward pass.
    x_smooth = np.array(x_filt, copy=True)
    for k in range(n - 2, -1, -1):
        # P_pred[k+1] = F P_filt[k] F.T + Q  (already stored)
        C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])

    return x_smooth[:, 0]
