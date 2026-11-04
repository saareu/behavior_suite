"""Resolve and pin an explicitly selected eligible S5 QC source.

Uses ``validate_s5_run_for_downstream`` — completed/completed_with_warnings with
live S1 digest match. Contours/masks are not persisted; QC replay loads the run
``settings_used.yaml`` and static ``cropped_background.png``. Never silently
selects the newest S5 run. Scientific trees remain read-only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.sync import SyncMap
from blob_features.config import (
    BlobFeaturesConfig,
    BlobSegmentationSettings,
    load_blob_features_config,
)
from blob_features.contracts import (
    BLOB_FEATURES_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    BlobFeaturesError,
)
from blob_features.runner import validate_s5_run_for_downstream
from overlay.contracts import (
    S5_SEGMENTATION_DRAW_IMPLEMENTATION_ID,
    S5_SEGMENTATION_REPLAY_IMPLEMENTATION_ID,
    OverlaySourceError,
)
from preprocess.sync_writer import load_prepared_sync_npz
from project.file_digest import sha256_file
from results_packaging.contracts import SYNC_SCHEMA_VERSION
from results_packaging.media_csv_export.video import VideoProbe, probe_video


@dataclass(frozen=True, slots=True)
class OverlayS5QcSource:
    """Pinned eligible S5 blob-features overlay source (explicit run only)."""

    s5_run_id: str
    s5_run_dir: Path
    status: str
    blob_features_path: Path
    blob_features_sha256: str
    settings_used_path: Path
    settings_used_sha256: str
    settings_fingerprint: str
    segmentation_settings: BlobSegmentationSettings
    config: BlobFeaturesConfig
    session_root: Path
    prepared_sync_path: Path
    prepared_sync_sha256: str
    prepared_video_path: Path
    prepared_video_sha256: str
    cropped_background_path: Path
    cropped_background_sha256: str
    prepare_meta_path: Path
    video_probe: VideoProbe
    sync: SyncMap
    sync_build_audit: dict[str, Any]
    prepared_frame_count: int
    prepared_fps: float
    run_meta: dict[str, Any]
    source_digests: dict[str, str | None]
    provenance: dict[str, Any] = field(default_factory=dict)

    def replay_disclosure(self) -> dict[str, Any]:
        runtime = collect_s5_replay_runtime_disclosure(
            self.segmentation_settings,
            run_meta=self.run_meta,
        )
        return {
            "geometry_source": "s5_segmentation_replay",
            "geometry_note": (
                "Masks and contours are not persisted in S5 artifacts. QC "
                "re-segments each prepared frame with the run's immutable "
                "settings_used.yaml and the pinned cropped_background.png via "
                f"{S5_SEGMENTATION_REPLAY_IMPLEMENTATION_ID}. Contours are QC "
                "replay, not stored truth polygons."
            ),
            "segmentation_implementation_id": S5_SEGMENTATION_REPLAY_IMPLEMENTATION_ID,
            "draw_implementation_id": S5_SEGMENTATION_DRAW_IMPLEMENTATION_ID,
            "settings_fingerprint": self.settings_fingerprint,
            "settings_used_path": str(self.settings_used_path),
            "settings_used_sha256": self.settings_used_sha256,
            "no_invented_blob_identities": True,
            "no_s3_s4_track_substitution": True,
            "no_contours_from_parquet_scalars": True,
            "parquet_role": (
                "blob_features.parquet is lineage/immutability only; scalars are "
                "not used to invent contours."
            ),
            "statefulness": (
                "Per-frame independent given (frame, background, settings); "
                "no adaptive background model or cross-frame segmentation state."
            ),
            "runtime": runtime,
        }


def resolve_overlay_s5_qc_source(
    *,
    s5_run_dir: Path | str,
) -> OverlayS5QcSource:
    """Open one explicitly selected eligible S5 run for segmentation QC export.

    Rejects incomplete / stale-source / missing-settings runs. Does not discover
    or fall back to another S5 run.
    """

    run_dir = Path(s5_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise OverlaySourceError(f"S5 run directory does not exist: {run_dir}")

    try:
        validated = validate_s5_run_for_downstream(run_dir)
    except BlobFeaturesError as exc:
        raise OverlaySourceError(
            f"S5 run is not eligible for overlay QC (downstream validation): {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"S5 run is not eligible for overlay QC (downstream validation): {exc}"
        ) from exc

    run_dir = Path(validated["s5_run_dir"]).resolve()
    session_root = Path(validated["session_root"]).resolve()
    meta_path = run_dir / RUN_META_FILENAME
    run_meta = _load_json_object(meta_path)
    status = str(validated["status"])

    settings_path = run_dir / SETTINGS_USED_FILENAME
    if not settings_path.is_file():
        raise OverlaySourceError(
            f"S5 settings_used.yaml missing; exact segmentation replay unavailable: "
            f"{settings_path}"
        )
    try:
        config = load_blob_features_config(settings_path)
    except BlobFeaturesError as exc:
        raise OverlaySourceError(
            f"S5 settings_used.yaml could not be loaded for faithful replay: {exc}"
        ) from exc
    segmentation = config.segmentation.validated()
    settings_fingerprint = segmentation_settings_fingerprint(segmentation)
    settings_sha = sha256_file(settings_path)
    # Fail closed when Multi-Otsu replay cannot run (no approximate threshold).
    _assert_replay_dependencies(segmentation)

    parquet_path = run_dir / BLOB_FEATURES_FILENAME
    if not parquet_path.is_file():
        raise OverlaySourceError(
            f"S5 blob_features.parquet missing for lineage pin: {parquet_path}"
        )
    parquet_sha = sha256_file(parquet_path)

    preprocess = session_root / "preprocess"
    prepared_video = preprocess / "prepared_video.mp4"
    prepared_sync = preprocess / "prepared_sync.npz"
    cropped_background = preprocess / "cropped_background.png"
    prepare_meta = preprocess / "prepare_meta.json"
    for path, label in (
        (prepared_video, "prepared_video.mp4"),
        (prepared_sync, "prepared_sync.npz"),
        (cropped_background, "cropped_background.png"),
        (prepare_meta, "prepare_meta.json"),
    ):
        if not path.is_file():
            raise OverlaySourceError(
                f"S1 {label} missing for S5 QC replay: {path}"
            )

    recorded = run_meta.get("source_s1")
    if not isinstance(recorded, Mapping):
        recorded = {}
    _assert_digest_match(
        label="prepared_video",
        path=prepared_video,
        recorded=_optional_text(recorded.get("prepared_video_sha256")),
        live=str(validated["prepared_video_sha256"]),
    )
    _assert_digest_match(
        label="prepared_sync",
        path=prepared_sync,
        recorded=_optional_text(recorded.get("prepared_sync_sha256")),
        live=str(validated["prepared_sync_sha256"]),
    )
    _assert_digest_match(
        label="cropped_background",
        path=cropped_background,
        recorded=_optional_text(recorded.get("cropped_background_sha256")),
        live=str(validated["cropped_background_sha256"]),
    )

    try:
        sync, sync_audit = _build_prepared_identity_sync(prepared_sync)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            "Authoritative prepared-frame timing could not be built from S1 "
            f"prepared_sync.npz (no dense 120 Hz mapping used): {exc}"
        ) from exc

    prepared_sync_sha = sha256_file(prepared_sync)
    video_sha = sha256_file(prepared_video)
    background_sha = sha256_file(cropped_background)
    try:
        video_probe = probe_video(prepared_video)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"Failed to probe S1 prepared video for S5 QC: {exc}"
        ) from exc

    prepared_frame_count = int(sync_audit["n_prepared"])
    prepared_fps = float(sync_audit["fps_header"])
    if int(video_probe.n_frames) > 0 and int(video_probe.n_frames) != prepared_frame_count:
        raise OverlaySourceError(
            "Prepared video frame count does not match prepared_sync frame domain: "
            f"video={video_probe.n_frames} sync={prepared_frame_count}."
        )

    source_digests: dict[str, str | None] = {
        "s5_blob_features_sha256": parquet_sha,
        "s5_settings_used_sha256": settings_sha,
        "s5_settings_fingerprint": settings_fingerprint,
        "s1_prepared_sync_sha256": prepared_sync_sha,
        "s1_prepared_video_sha256": video_sha,
        "s1_cropped_background_sha256": background_sha,
    }

    return OverlayS5QcSource(
        s5_run_id=run_dir.name,
        s5_run_dir=run_dir,
        status=status,
        blob_features_path=parquet_path,
        blob_features_sha256=parquet_sha,
        settings_used_path=settings_path,
        settings_used_sha256=settings_sha,
        settings_fingerprint=settings_fingerprint,
        segmentation_settings=segmentation,
        config=config,
        session_root=session_root,
        prepared_sync_path=prepared_sync,
        prepared_sync_sha256=prepared_sync_sha,
        prepared_video_path=prepared_video,
        prepared_video_sha256=video_sha,
        cropped_background_path=cropped_background,
        cropped_background_sha256=background_sha,
        prepare_meta_path=prepare_meta,
        video_probe=video_probe,
        sync=sync,
        sync_build_audit=sync_audit,
        prepared_frame_count=prepared_frame_count,
        prepared_fps=prepared_fps,
        run_meta=dict(run_meta),
        source_digests=source_digests,
        provenance={
            "source_kind": "s5_eligible_blob_features",
            "s5_run_id": run_dir.name,
            "s5_run_dir": str(run_dir),
            "status": status,
            "eligibility": "validate_s5_run_for_downstream",
            "prepared_sync": str(prepared_sync),
            "prepared_video": str(prepared_video),
            "cropped_background": str(cropped_background),
            "timing_authority": "s1_prepared_frame_identity",
            "timing_note": (
                "S5 contour QC uses S1 prepared-frame indices and prepared-video "
                "FPS. SyncMap is a 1:1 prepared-identity map for compositor reuse "
                "— not S4 dense 120 Hz duplication. Dense parquet frame_idx is "
                "never used as a video seek index."
            ),
            "segmentation_implementation_id": S5_SEGMENTATION_REPLAY_IMPLEMENTATION_ID,
            "draw_implementation_id": S5_SEGMENTATION_DRAW_IMPLEMENTATION_ID,
            "settings_fingerprint": settings_fingerprint,
            "replay": {
                "geometry_source": "s5_segmentation_replay",
                "masks_persisted": False,
                "contours_persisted": False,
                "requires_background": True,
                "requires_settings_used": True,
            },
        },
    )


def verify_overlay_s5_qc_source_integrity(source: OverlayS5QcSource) -> None:
    """Recheck S5 eligibility and digests before publication."""

    try:
        validate_s5_run_for_downstream(source.s5_run_dir)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"S5 eligibility failed during integrity recheck: {exc}"
        ) from exc

    _assert_replay_dependencies(source.segmentation_settings)

    live: dict[str, str | None] = {
        "s5_blob_features_sha256": (
            sha256_file(source.blob_features_path)
            if source.blob_features_path.is_file()
            else None
        ),
        "s5_settings_used_sha256": (
            sha256_file(source.settings_used_path)
            if source.settings_used_path.is_file()
            else None
        ),
        "s1_prepared_sync_sha256": (
            sha256_file(source.prepared_sync_path)
            if source.prepared_sync_path.is_file()
            else None
        ),
        "s1_prepared_video_sha256": (
            sha256_file(source.prepared_video_path)
            if source.prepared_video_path.is_file()
            else None
        ),
        "s1_cropped_background_sha256": (
            sha256_file(source.cropped_background_path)
            if source.cropped_background_path.is_file()
            else None
        ),
    }
    for key, expected in source.source_digests.items():
        if expected is None or key not in live:
            continue
        actual = live.get(key)
        if actual is None or str(actual) != str(expected):
            raise OverlaySourceError(
                f"Overlay S5 QC source digest mismatch for {key}: "
                f"pinned={expected!r} live={actual!r}."
            )

    # Settings fingerprint must still match the immutable YAML snapshot.
    try:
        live_config = load_blob_features_config(source.settings_used_path)
        live_fp = segmentation_settings_fingerprint(live_config.segmentation)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"S5 settings fingerprint revalidation failed: {exc}"
        ) from exc
    if live_fp != source.settings_fingerprint:
        raise OverlaySourceError(
            "S5 settings fingerprint mismatch during integrity recheck: "
            f"pinned={source.settings_fingerprint!r} live={live_fp!r}."
        )


def segmentation_settings_fingerprint(settings: BlobSegmentationSettings) -> str:
    """Stable SHA256 over validated segmentation knobs (not execution/temporal)."""

    validated = settings.validated()
    payload = json.dumps(asdict(validated), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_s5_replay_runtime_disclosure(
    settings: BlobSegmentationSettings,
    *,
    run_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Disclose current replay runtime deps without claiming historical equivalence.

    S5 ``run_meta.json`` historically does not record OpenCV / NumPy /
    scikit-image / Python versions. Contour QC therefore discloses the *current*
    interpreter environment and refuses Multi-Otsu when scikit-image is absent.
    """

    validated = settings.validated()
    multi_otsu_required = int(validated.blob_threshold) == -2
    skimage_version, skimage_available, skimage_error = _probe_skimage()
    historical_recorded = _historical_environment_recorded(run_meta)
    return {
        "python_version": _python_version_text(),
        "opencv_version": _package_version("cv2"),
        "numpy_version": _package_version("numpy"),
        "skimage_version": skimage_version,
        "skimage_available": skimage_available,
        "skimage_import_error": skimage_error,
        "multi_otsu_required": multi_otsu_required,
        "blob_threshold": int(validated.blob_threshold),
        "historical_environment_recorded": historical_recorded,
        "historical_environment_equivalence_claimed": False,
        "note": (
            "Runtime versions are the current QC host environment. S5 run_meta "
            "does not record the original extraction environment; exact "
            "bit-for-bit historical environment equivalence is not claimed."
        ),
    }


def _assert_replay_dependencies(settings: BlobSegmentationSettings) -> None:
    """Refuse approximate replay when required Multi-Otsu deps are missing."""

    validated = settings.validated()
    if int(validated.blob_threshold) != -2:
        return
    _version, available, error = _probe_skimage()
    if available:
        return
    detail = f" ({error})" if error else ""
    raise OverlaySourceError(
        "S5 Multi-Otsu replay (blob_threshold=-2) requires scikit-image; "
        f"refusing approximate contour overlay{detail}."
    )


def _probe_skimage() -> tuple[str | None, bool, str | None]:
    try:
        import skimage
    except Exception as exc:  # noqa: BLE001
        return None, False, f"{type(exc).__name__}: {exc}"
    version = getattr(skimage, "__version__", None)
    return (str(version) if version is not None else None), True, None


def _package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:  # noqa: BLE001
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _python_version_text() -> str:
    import sys

    return sys.version.split()[0]


def _historical_environment_recorded(run_meta: Mapping[str, Any] | None) -> bool:
    if not isinstance(run_meta, Mapping):
        return False
    for key in ("environment", "dependencies", "versions", "runtime"):
        value = run_meta.get(key)
        if isinstance(value, Mapping) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _build_prepared_identity_sync(
    prepared_sync_path: Path,
) -> tuple[SyncMap, dict[str, Any]]:
    sync_data = load_prepared_sync_npz(prepared_sync_path)
    n = int(sync_data.frame_count_used_for_sleap)
    if n <= 0:
        raise OverlaySourceError("prepared_sync.npz reports non-positive frame count.")
    prepared_idx = np.asarray(sync_data.prepared_frame_idx, dtype=np.int64)[:n]
    prepared_time = np.asarray(sync_data.prepared_time_sec, dtype=np.float64)[:n]
    if prepared_idx.shape[0] != n or prepared_time.shape[0] != n:
        raise OverlaySourceError(
            "prepared_sync.npz prepared arrays shorter than frame_count_used_for_sleap."
        )
    expected = np.arange(n, dtype=np.int64)
    if not np.array_equal(prepared_idx, expected):
        raise OverlaySourceError(
            "prepared_sync.npz prepared_frame_idx must equal contiguous 0..N-1."
        )
    fps = float(sync_data.fps_header)
    if not np.isfinite(fps) or fps <= 0:
        raise OverlaySourceError(
            f"prepared_sync.npz fps_header must be positive; got {fps}."
        )

    table = pd.DataFrame(
        {
            "schema_version": SYNC_SCHEMA_VERSION,
            "sample_idx": prepared_idx,
            "dense_frame_idx": prepared_idx,
            "time_sec": prepared_time,
            "prepared_frame_idx": prepared_idx,
            "prepared_time_sec": prepared_time,
            "is_exact_source_sample": np.ones(n, dtype=bool),
            "timing_residual_sec": np.zeros(n, dtype=np.float64),
            "raw_decode_frame_idx": np.asarray(
                sync_data.raw_decode_frame_idx, dtype=np.int64
            )[:n],
        }
    )
    sync = SyncMap.from_sync_index(table, fps=fps)
    audit = {
        "kind": "s1_prepared_frame_identity",
        "n_prepared": n,
        "fps_header": fps,
        "max_many_to_one": 1,
        "dense_120hz": False,
        "note": (
            "Identity prepared↔sample map for compositor timing reuse; "
            "not an S4 dense 120 Hz timeline."
        ),
    }
    return sync, audit


def _assert_digest_match(
    *,
    label: str,
    path: Path,
    recorded: str | None,
    live: str,
) -> None:
    if recorded is None:
        return
    if recorded.lower() != live.lower():
        raise OverlaySourceError(
            f"S5 source S1 {label} digest mismatch for {path}: "
            f"recorded={recorded!r} live={live!r}."
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(f"Failed to read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OverlaySourceError(f"Expected JSON object at {path}.")
    return payload


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


__all__ = [
    "OverlayS5QcSource",
    "collect_s5_replay_runtime_disclosure",
    "resolve_overlay_s5_qc_source",
    "segmentation_settings_fingerprint",
    "verify_overlay_s5_qc_source_integrity",
]
