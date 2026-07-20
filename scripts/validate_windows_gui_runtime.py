"""Post-install import validation for the supported Windows GUI environment."""

from __future__ import annotations

import importlib
import sys

EXPECTED_PYSIDE_VERSION = "6.11.1"
EXPECTED_SLEAP_IO_VERSION = "0.8.0"

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

APPLICATION_MODULES = (
    "cli.preprocess",
    "preprocess.service",
    "pose_inference.runner",
    "ui.main_window",
)


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

    if failures:
        print("Windows GUI runtime validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"PySide6 import/version: ok ({observed_version})")
    print(f"sleap-io import/version: ok ({observed_sleap_io_version})")
    print(f"Application dependency imports: ok ({len(DEPENDENCY_MODULES)} modules)")
    print(f"S2 artifact module imports: ok ({len(S2_ARTIFACT_MODULES)} modules)")
    print(f"Application module imports: ok ({len(APPLICATION_MODULES)} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
