"""Headless Subsystem 09 runners (Sprint 1–3).

Scientific transformations stay independent of GUI code. Sprint 1–2 produce
in-memory package results. Sprint 3 immutable publication lives in
``results_packaging.publication`` and is re-exported here for a stable API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import QC_FILENAME
from results_packaging.bout_mapping import map_bout_boundaries_to_prepared_frames
from results_packaging.bouts import assert_lossless_bout_copy, package_behavior_bouts
from results_packaging.config import (
    ResultsPackagingConfig,
    ReviewEmitV1Defaults,
    config_from_mapping,
    load_results_packaging_config,
)
from results_packaging.contracts import (
    DEFAULT_PROFILE_ID,
    REVIEW_EMIT_PROFILE_ID,
    IntakeError,
    ResultsPackagingError,
)
from results_packaging.dependencies import build_dependency_manifest
from results_packaging.eligibility import validate_s8_source_for_packaging
from results_packaging.ethogram_summary import build_ethogram_summary
from results_packaging.export_profiles import (
    ReviewEmitV1Params,
    apply_review_emit_v1,
)
from results_packaging.results import Sprint1PackagingResult, Sprint2PackagingResult
from results_packaging.sync_index import build_sync_index_from_prepared_sync


@dataclass(frozen=True, slots=True)
class ResultsPackagingRequest:
    """Explicit packaging / publication request (Sprint 1–3)."""

    session_root: Path
    s8_run_dir: Path | None = None
    config: ResultsPackagingConfig | None = None
    settings_overrides: Mapping[str, Any] | None = None
    package_run_id: str | None = None
    # When True (Sprint 2/3 path), build ethogram/bouts/profiles.
    # Sprint 1 entry points leave this False.
    build_package_tables: bool = False
    # Sprint 3 publication fields.
    output_root: Path | None = None
    timestamp: str | None = None
    run_purpose: str = "development"


def run_results_packaging_sprint1(
    request: ResultsPackagingRequest,
) -> Sprint1PackagingResult:
    """Validate S8 source, build sync index, prepare bout-boundary mappings.

    Returns in-memory diagnostics and dependency identities. Does not write an
    S9 run directory and does not modify upstream S1–S8 artifacts.
    """

    core = _run_core(replace(request, build_package_tables=False))
    return Sprint1PackagingResult(
        source=core["source"],
        sync=core["sync"],
        bout_boundary_map=core["bout_boundary_map"],
        dependencies=core["dependencies"],
        settings_used=core["settings_used"],
        session_root=core["session_root"],
        warnings=core["warnings"],
        diagnostics=core["diagnostics_sprint1"],
    )


def run_results_packaging_sprint2(
    request: ResultsPackagingRequest,
) -> Sprint2PackagingResult:
    """Build the in-memory S9 package: sync, full bouts, summary, optional profiles.

    Does not publish an immutable S9 run and does not modify upstream artifacts.
    ``review_emit_v1`` runs only when explicitly listed in
    ``execution.selected_export_profiles``.
    """

    core = _run_core(replace(request, build_package_tables=True))
    return Sprint2PackagingResult(
        source=core["source"],
        sync=core["sync"],
        bout_boundary_map=core["bout_boundary_map"],
        behavior_bouts=core["behavior_bouts"],
        ethogram_summary=core["ethogram_summary"],
        export_profiles=core["export_profiles"],
        dependencies=core["dependencies"],
        settings_used=core["settings_used"],
        session_root=core["session_root"],
        warnings=core["warnings"],
        diagnostics=core["diagnostics_sprint2"],
        package_run_id=core["package_run_id"],
    )


def run_sprint1_from_session(
    session_root: Path | str,
    *,
    s8_run_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    config: ResultsPackagingConfig | None = None,
) -> Sprint1PackagingResult:
    """Convenience wrapper around ``run_results_packaging_sprint1``."""

    resolved_config = config
    if resolved_config is None and config_path is not None:
        resolved_config = load_results_packaging_config(config_path)
    return run_results_packaging_sprint1(
        ResultsPackagingRequest(
            session_root=Path(session_root),
            s8_run_dir=Path(s8_run_dir) if s8_run_dir is not None else None,
            config=resolved_config,
        )
    )


def run_sprint2_from_session(
    session_root: Path | str,
    *,
    s8_run_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    config: ResultsPackagingConfig | None = None,
    package_run_id: str | None = None,
) -> Sprint2PackagingResult:
    """Convenience wrapper around ``run_results_packaging_sprint2``."""

    resolved_config = config
    if resolved_config is None and config_path is not None:
        resolved_config = load_results_packaging_config(config_path)
    return run_results_packaging_sprint2(
        ResultsPackagingRequest(
            session_root=Path(session_root),
            s8_run_dir=Path(s8_run_dir) if s8_run_dir is not None else None,
            config=resolved_config,
            package_run_id=package_run_id,
            build_package_tables=True,
        )
    )


def _run_core(request: ResultsPackagingRequest) -> dict[str, Any]:
    session_root = Path(request.session_root).expanduser().resolve()
    if not session_root.is_dir():
        raise IntakeError(f"Session root does not exist: {session_root}")

    config = _resolve_config(request)
    settings_used = config.to_settings_dict()
    if request.s8_run_dir is not None:
        settings_used["selected_s8_run_dir"] = str(
            Path(request.s8_run_dir).expanduser().resolve()
        )
    else:
        settings_used["selected_s8_run_dir"] = None
        settings_used["selection_mode"] = "latest_eligible"

    package_run_id = request.package_run_id
    settings_used["package_run_id"] = package_run_id

    source = validate_s8_source_for_packaging(
        session_root,
        s8_run_dir=request.s8_run_dir,
        require_prepared_video=config.execution.require_prepared_video,
    )

    sync = build_sync_index_from_prepared_sync(
        source.sync_video.prepared_sync_path,
        s8_unique_dense_times=source.unique_dense_times,
        package_run_id=package_run_id,
        target_fps=config.fps,
    )

    s8_bouts = pd.read_parquet(source.bouts_path)
    bout_map = map_bout_boundaries_to_prepared_frames(
        s8_bouts, sync.sync_index, fps=config.fps
    )

    dependencies = build_dependency_manifest(
        source,
        package_mode=config.execution.package_mode,
        sync_construction_method=config.execution.sync_construction_method,
        hash_prepared_video=config.execution.hash_prepared_video,
    )

    warnings = list(
        dict.fromkeys([*source.warnings, *sync.warnings, *dependencies.notes])
    )
    diagnostics_sprint1 = {
        "source": source.to_dict(),
        "sync_audit": sync.audit_dict(),
        "bout_mapping": {
            "n_bouts_mapped": int(len(bout_map)),
            "n_bouts_source": int(source.n_bouts),
            "duration_frames_unchanged": True,
        },
        "published": False,
    }
    if int(len(bout_map)) != int(source.n_bouts):
        raise ResultsPackagingError(
            "Bout-boundary mapping row count differs from S8 bout count."
        )

    out: dict[str, Any] = {
        "source": source,
        "sync": sync,
        "bout_boundary_map": bout_map,
        "dependencies": dependencies,
        "settings_used": settings_used,
        "session_root": session_root,
        "package_run_id": package_run_id,
        "diagnostics_sprint1": diagnostics_sprint1,
        "behavior_bouts": None,
        "ethogram_summary": None,
        "export_profiles": {},
        "diagnostics_sprint2": {},
        "warnings": tuple(warnings),
    }

    if not request.build_package_tables:
        return out

    packaged = package_behavior_bouts(
        s8_bouts,
        sync.sync_index,
        source_s8_run_id=source.s8_run_id,
        package_run_id=package_run_id,
        fps=config.fps,
    )
    assert_lossless_bout_copy(s8_bouts, packaged)

    states = pd.read_parquet(source.states_path)
    s8_qc_metrics = _load_s8_qc_metrics(source.s8_run_dir)
    ethogram = build_ethogram_summary(
        packaged_bouts=packaged,
        states=states,
        sync_index=sync.sync_index,
        fps=config.fps,
        s8_qc_metrics=s8_qc_metrics,
    )

    export_profiles: dict[str, Any] = {}
    selected = tuple(config.execution.selected_export_profiles)
    if REVIEW_EMIT_PROFILE_ID in selected:
        profile = apply_review_emit_v1(
            packaged,
            params=_review_params_from_defaults(config.review_emit_v1_defaults),
        )
        export_profiles[REVIEW_EMIT_PROFILE_ID] = profile
        warnings.extend(profile.warnings)

    # Default package never auto-applies review_emit_v1.
    if not selected:
        settings_used["export_profiles_applied"] = []
    else:
        settings_used["export_profiles_applied"] = list(selected)
    settings_used["default_profile_id"] = DEFAULT_PROFILE_ID

    channel_counts = {
        str(k): int(v)
        for k, v in packaged["channel"].value_counts().sort_index().items()
    }
    n_short_exclusive = int(
        (
            (packaged["channel"].astype(str) == "exclusive")
            & (packaged["bout_duration_frames"] < 10)
        ).sum()
    )
    diagnostics_sprint2 = {
        **diagnostics_sprint1,
        "bout_packaging": {
            "n_bouts_packaged": int(len(packaged)),
            "n_bouts_source": int(len(s8_bouts)),
            "lossless": True,
            "channel_counts": channel_counts,
            "n_exclusive_duration_lt_10": n_short_exclusive,
            "prepared_navigation_annotations": True,
            "durations_not_recomputed_from_prepared_frames": True,
        },
        "ethogram_summary": {
            "n_rows": int(len(ethogram)),
            "summary_kinds": sorted(
                ethogram["summary_kind"].astype(str).unique().tolist()
            )
            if not ethogram.empty
            else [],
        },
        "export_profiles": {
            profile_id: result.to_manifest_dict()
            for profile_id, result in export_profiles.items()
        },
        "review_emit_v1_auto_applied": False,
        "published": False,
    }

    out.update(
        {
            "behavior_bouts": packaged,
            "ethogram_summary": ethogram,
            "export_profiles": export_profiles,
            "diagnostics_sprint2": diagnostics_sprint2,
            "warnings": tuple(dict.fromkeys(warnings)),
        }
    )
    return out


def _review_params_from_defaults(defaults: ReviewEmitV1Defaults) -> ReviewEmitV1Params:
    return ReviewEmitV1Params(
        min_emit_bout_frames=int(defaults.min_emit_bout_frames),
        min_emit_bout_seconds=float(defaults.min_emit_bout_seconds),
        target_fps=float(defaults.target_fps),
        include_contact=bool(defaults.include_contact),
        include_engaged=bool(defaults.include_engaged),
        include_side_by_side=bool(defaults.include_side_by_side),
    )


def _load_s8_qc_metrics(s8_run_dir: Path) -> dict[str, Any] | None:
    qc_path = Path(s8_run_dir) / QC_FILENAME
    if not qc_path.is_file():
        return None
    try:
        payload = json.loads(qc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    return dict(metrics) if isinstance(metrics, Mapping) else None


def _resolve_config(request: ResultsPackagingRequest) -> ResultsPackagingConfig:
    if request.config is not None and request.settings_overrides:
        raise IntakeError("Pass either config or settings_overrides, not both.")
    if request.config is not None:
        return request.config.validated()
    if request.settings_overrides is not None:
        return config_from_mapping(dict(request.settings_overrides))
    return load_results_packaging_config(None)


# Sprint 3 publication API (lazy imports avoid circular load with publication.py).
def run_results_packaging_publication(*args: Any, **kwargs: Any) -> Any:
    from results_packaging.publication import (
        run_results_packaging_publication as _impl,
    )

    return _impl(*args, **kwargs)


def run_publication_from_session(*args: Any, **kwargs: Any) -> Any:
    from results_packaging.publication import run_publication_from_session as _impl

    return _impl(*args, **kwargs)


def is_s9_run_eligible_for_downstream(*args: Any, **kwargs: Any) -> bool:
    from results_packaging.publication import is_s9_run_eligible_for_downstream as _impl

    return bool(_impl(*args, **kwargs))


def validate_s9_run_for_downstream(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from results_packaging.publication import validate_s9_run_for_downstream as _impl

    return dict(_impl(*args, **kwargs))


def reopen_results_package(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from results_packaging.publication import reopen_results_package as _impl

    return dict(_impl(*args, **kwargs))
