"""Nonblocking doctor advisories for optional S2/S7 inference dependencies.

Hard launch gate remains FFmpeg capability preflight. Missing sleap-nn, S7
runtime libraries, or model registries must warn operators without blocking GUI
startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from behavioral_inference.contracts import (
    BUNDLE_INCOMPLETE,
    BUNDLE_SUPPORTED,
    REGISTRY_OK,
    REGISTRY_UNAVAILABLE,
    S7_RUNTIME_INSTALL_HINT,
)
from behavioral_inference.gui_persistence import (
    SettingsStore,
    load_persisted_registry_root,
    suggest_lab_model_registry,
)
from behavioral_inference.loader import probe_runtime_dependencies
from behavioral_inference.registry import discover_model_registry
from combined_features.config import ModelRegistrySettings
from pose_inference.runner import SleapNnProbeResult, probe_sleap_nn_runtime


@dataclass(frozen=True)
class DoctorAdvisory:
    """One advisory line for doctor output."""

    level: str  # "ok" | "warn" | "info"
    message: str


def _try_load_gui_settings() -> SettingsStore | None:
    """Load persisted GUI settings when PySide6 is importable; else None."""

    try:
        from PySide6.QtCore import QSettings  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001 - advisory path; Qt optional at doctor time
        return None
    settings: SettingsStore = QSettings("BehaviorSuite", "BehaviorSuite")
    return settings


def collect_sleap_nn_advisories(
    *,
    sleap_probe: SleapNnProbeResult | None = None,
) -> list[DoctorAdvisory]:
    """Advise on sleap-nn 0.3.x availability (never a hard failure)."""

    probe = sleap_probe if sleap_probe is not None else probe_sleap_nn_runtime()
    if probe.compatible:
        return [DoctorAdvisory(level="ok", message=probe.message)]
    return [
        DoctorAdvisory(level="warn", message=probe.message),
        DoctorAdvisory(
            level="info",
            message=(
                "S2 inference models are operator-selected lab paths and are not "
                "verified by doctor; configure them in the S2 GUI when ready."
            ),
        ),
    ]


def collect_s7_dependency_advisories() -> list[DoctorAdvisory]:
    """Advise on optional S7 Python runtime libraries."""

    status = probe_runtime_dependencies()
    if status.available:
        version_bits = ", ".join(
            f"{name}={version or 'unknown'}"
            for name, version in sorted(status.versions.items())
        )
        return [
            DoctorAdvisory(
                level="ok",
                message=f"S7 runtime dependencies: available ({version_bits})",
            )
        ]
    missing = ", ".join(status.missing) if status.missing else "unknown"
    return [
        DoctorAdvisory(
            level="warn",
            message=(
                f"S7 not ready: missing runtime dependencies [{missing}]. "
                f"{S7_RUNTIME_INSTALL_HINT}"
            ),
        )
    ]


def collect_s7_registry_advisories(
    *,
    settings: SettingsStore | None = None,
    repo_root: Path | str | None = None,
) -> list[DoctorAdvisory]:
    """Advise on persisted/suggested S7 model registry discovery."""

    advisories: list[DoctorAdvisory] = []
    store = settings if settings is not None else _try_load_gui_settings()
    persisted = load_persisted_registry_root(store)
    suggested = suggest_lab_model_registry(repo_root)

    root: Path | None = None
    root_source: str
    if persisted is not None and persisted.is_dir():
        root = persisted
        root_source = "persisted GUI path"
    elif suggested is not None:
        root = suggested
        root_source = "suggested repository models/"
        if persisted is None:
            advisories.append(
                DoctorAdvisory(
                    level="info",
                    message=(
                        "S7 registry: no persisted GUI path yet; "
                        f"suggested browse start: {suggested}"
                    ),
                )
            )
    elif persisted is not None:
        advisories.append(
            DoctorAdvisory(
                level="warn",
                message=(
                    "S7 registry: persisted path is unresolved or not a directory: "
                    f"{persisted}. Browse again in Subsystem 7 (prefer models/). "
                    "Historical legacy/models digests may block archival package "
                    "eligibility until remapped."
                ),
            )
        )
        return advisories
    else:
        advisories.append(
            DoctorAdvisory(
                level="warn",
                message=(
                    "S7 registry: unavailable — browse to a trusted models/ tree "
                    "in Subsystem 7 (dyadic/v1 and focal/v1). GUI can launch without "
                    "this; S7 Run stays blocked until configured."
                ),
            )
        )
        return advisories

    snapshot = discover_model_registry(ModelRegistrySettings(root=root))
    if snapshot.status == REGISTRY_UNAVAILABLE:
        notes = "; ".join(snapshot.notes) if snapshot.notes else "unavailable"
        advisories.append(
            DoctorAdvisory(
                level="warn",
                message=(
                    f"S7 registry: unavailable via {root_source} ({root}): {notes}"
                ),
            )
        )
        return advisories

    incomplete = [
        b.detector_name
        for b in snapshot.bundles
        if b.support == BUNDLE_INCOMPLETE and b.detector_name != "contact"
    ]
    sha_mismatch = [
        b.detector_name
        for b in snapshot.bundles
        if b.sha256_matches_expected is False
    ]
    missing_model_files = [
        b.detector_name
        for b in snapshot.bundles
        if b.support in (BUNDLE_SUPPORTED, BUNDLE_INCOMPLETE)
        and b.model_path is None
        and b.detector_name != "contact"
    ]

    supported = sum(1 for b in snapshot.bundles if b.support == BUNDLE_SUPPORTED)
    loadable = snapshot.loadable_bundles()
    # discover_model_registry marks status=partial whenever notes exist, including
    # the always-reported obsolete standalone "contact" detector. Treat that alone
    # as healthy for doctor advisories when loadable bundles are present.
    obsolete_contact_only_notes = snapshot.notes and all(
        "obsolete" in note.lower() and "contact" in note.lower()
        for note in snapshot.notes
    )
    healthy = (
        bool(loadable)
        and not incomplete
        and not sha_mismatch
        and not missing_model_files
        and (
            snapshot.status == REGISTRY_OK
            or obsolete_contact_only_notes
        )
    )
    if healthy:
        advisories.append(
            DoctorAdvisory(
                level="ok",
                message=(
                    f"S7 registry: ready via {root_source} ({root}); "
                    f"{supported} supported bundle(s)"
                ),
            )
        )
    else:
        detail_parts = [f"status={snapshot.status}"]
        if incomplete:
            detail_parts.append(f"incomplete={','.join(incomplete)}")
        if missing_model_files:
            detail_parts.append(f"missing_models={','.join(missing_model_files)}")
        if sha_mismatch:
            detail_parts.append(f"sha256_mismatch={','.join(sha_mismatch)}")
        material_notes = [
            note
            for note in snapshot.notes
            if not ("obsolete" in note.lower() and "contact" in note.lower())
        ]
        if material_notes:
            detail_parts.append("; ".join(material_notes))
        advisories.append(
            DoctorAdvisory(
                level="warn",
                message=(
                    f"S7 registry: partial/incomplete via {root_source} ({root}) — "
                    + "; ".join(detail_parts)
                ),
            )
        )

    # Historical path footgun when both trees exist locally.
    if root is not None:
        legacy = root.parent / "legacy" / "models"
        if root.name.lower() == "models" and legacy.is_dir():
            advisories.append(
                DoctorAdvisory(
                    level="info",
                    message=(
                        f"Historical path note: {legacy} exists beside {root}. "
                        "Prefer new runs with registry root models/; remap "
                        "legacy/models digests only for archival eligibility."
                    ),
                )
            )

    return advisories


def collect_inference_advisories(
    *,
    sleap_probe: SleapNnProbeResult | None = None,
    settings: SettingsStore | None = None,
    repo_root: Path | str | None = None,
) -> list[DoctorAdvisory]:
    """Collect all nonblocking S2/S7 doctor advisories."""

    lines: list[DoctorAdvisory] = []
    lines.extend(collect_sleap_nn_advisories(sleap_probe=sleap_probe))
    lines.extend(collect_s7_dependency_advisories())
    lines.extend(
        collect_s7_registry_advisories(settings=settings, repo_root=repo_root)
    )
    return lines


def format_inference_advisories(advisories: list[DoctorAdvisory]) -> str:
    """Format advisories for doctor stdout."""

    if not advisories:
        return "Inference advisories: none"
    lines = ["Inference advisories (nonblocking; GUI may still launch):"]
    for item in advisories:
        prefix = {
            "ok": "OK",
            "warn": "WARN",
            "info": "INFO",
        }.get(item.level, item.level.upper())
        lines.append(f"- [{prefix}] {item.message}")
    return "\n".join(lines)
