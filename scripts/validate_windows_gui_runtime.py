"""Post-install import validation for the supported Windows GUI environment."""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version

EXPECTED_PYSIDE_VERSION = "6.11.1"
EXPECTED_SLEAP_IO_VERSION = "0.8.0"

# Pins must stay aligned with pyproject.toml [project.optional-dependencies].s7
# and docs/subsystem_07/sprint_1_model_runtime.md.
S7_PACKAGE_CONSTRAINTS = (
    ("xgboost", "3.2.0", "3.3"),
    ("scikit-learn", "1.6.1", "1.7"),
    ("joblib", "1.3", "2"),
)

DEPENDENCY_MODULES = (
    "numpy",
    "cv2",
    "scipy",
    "h5py",
    "pandas",
    "pyarrow",
    "pydantic",
    "yaml",
    "typer",
    "rich",
    "platformdirs",
)

QT_MODULES = (
    "shiboken6",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
)

S2_DEPENDENCY_MODULES = (
    "sleap_io",
)

S2_ARTIFACT_MODULES = (
    "pose_inference.parquet_export",
    "pose_inference.pose_qc",
    "pose_inference.overlay",
)

S7_DEPENDENCY_MODULES = (
    "xgboost",
    "sklearn",
    "joblib",
)

APPLICATION_MODULES = (
    "cli.preprocess",
    "preprocess.service",
    "pose_inference.runner",
    "behavioral_inference.loader",
    "ui.main_window",
)


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in raw.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_in_range(observed: str, minimum: str, exclusive_maximum: str) -> bool:
    current = _parse_version_tuple(observed)
    low = _parse_version_tuple(minimum)
    high = _parse_version_tuple(exclusive_maximum)
    if not current or not low or not high:
        return False
    return low <= current < high


def _check_s7_package_versions(failures: list[str]) -> list[str]:
    observed_lines: list[str] = []
    for package_name, minimum, exclusive_maximum in S7_PACKAGE_CONSTRAINTS:
        try:
            observed = version(package_name)
        except PackageNotFoundError:
            failures.append(f"{package_name}: PackageNotFoundError: not installed")
            continue
        observed_lines.append(f"{package_name}=={observed}")
        if not _version_in_range(observed, minimum, exclusive_maximum):
            failures.append(
                f"{package_name} version mismatch: expected >={minimum},<{exclusive_maximum}, "
                f"found {observed}"
            )
    return observed_lines


def main() -> int:
    """Import the installed runtime surface and require the pinned Qt version."""

    print(f"Python executable: {sys.executable}")
    failures: list[str] = []
    imported: dict[str, object] = {}
    runtime_modules = (
        *DEPENDENCY_MODULES,
        *QT_MODULES,
        *S2_DEPENDENCY_MODULES,
        *S2_ARTIFACT_MODULES,
        *S7_DEPENDENCY_MODULES,
        *APPLICATION_MODULES,
    )
    for module_name in runtime_modules:
        try:
            imported[module_name] = importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    pyside = imported.get("PySide6")
    observed_version = getattr(pyside, "__version__", None)
    if pyside is not None and observed_version != EXPECTED_PYSIDE_VERSION:
        failures.append(
            "PySide6 version mismatch: "
            f"expected {EXPECTED_PYSIDE_VERSION}, found {observed_version or 'unknown'}"
        )

    sleap_io = imported.get("sleap_io")
    observed_sleap_io_version = getattr(sleap_io, "__version__", None)
    if sleap_io is not None and observed_sleap_io_version != EXPECTED_SLEAP_IO_VERSION:
        failures.append(
            "sleap-io version mismatch: "
            f"expected {EXPECTED_SLEAP_IO_VERSION}, "
            f"found {observed_sleap_io_version or 'unknown'}"
        )

    s7_versions = _check_s7_package_versions(failures)

    if failures:
        print("Windows GUI runtime validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"PySide6 import/version: ok ({observed_version})")
    print(f"sleap-io import/version: ok ({observed_sleap_io_version})")
    print(f"S7 runtime packages: ok ({', '.join(s7_versions)})")
    print(f"Application dependency imports: ok ({len(DEPENDENCY_MODULES)} modules)")
    print(f"S2 artifact module imports: ok ({len(S2_ARTIFACT_MODULES)} modules)")
    print(f"S7 dependency imports: ok ({len(S7_DEPENDENCY_MODULES)} modules)")
    print(f"Application module imports: ok ({len(APPLICATION_MODULES)} modules)")
    print(
        "Note: S7 runtime libraries are installed; trusted model bundles must still "
        "be configured in the GUI model-registry path."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
