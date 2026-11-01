"""Atomic persistence and safe loading for workspace_catalog_v1."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from workspace.errors import (
    WorkspaceCatalogError,
    WorkspaceCatalogVersionError,
    WorkspaceDuplicateError,
    WorkspaceValidationError,
)
from workspace.models import (
    CATALOG_BACKUP_SUFFIX,
    COMPATIBLE_READ_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    SessionRecord,
    WorkspaceCatalog,
    catalog_path_for_root,
    path_to_storage,
    utc_now_iso,
    with_updated_path_status,
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def catalog_to_jsonable(catalog: WorkspaceCatalog) -> dict[str, Any]:
    """Idempotent JSON-ready dict (sorted keys at dump time; stable model dump)."""

    payload = catalog.model_dump(mode="json")
    # Ensure schema_version is exactly the writer version.
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def serialize_catalog(catalog: WorkspaceCatalog) -> str:
    """Deterministic catalog serialization."""

    return json.dumps(catalog_to_jsonable(catalog), indent=2, sort_keys=True) + "\n"


def assert_supported_schema_version(schema_version: object) -> str:
    if not isinstance(schema_version, str):
        raise WorkspaceCatalogVersionError(
            f"Catalog schema_version must be a string, got {type(schema_version)!r}."
        )
    if schema_version not in COMPATIBLE_READ_SCHEMA_VERSIONS:
        raise WorkspaceCatalogVersionError(
            f"Unsupported workspace catalog schema_version: {schema_version!r}. "
            f"Supported: {sorted(COMPATIBLE_READ_SCHEMA_VERSIONS)}"
        )
    return schema_version


def parse_catalog_payload(payload: dict[str, Any]) -> WorkspaceCatalog:
    """Validate a raw dict into a WorkspaceCatalog."""

    assert_supported_schema_version(payload.get("schema_version"))
    try:
        catalog = WorkspaceCatalog.model_validate(payload)
    except ValidationError as exc:
        raise WorkspaceValidationError(f"Invalid workspace catalog payload: {exc}") from exc
    except (WorkspaceDuplicateError, WorkspaceValidationError):
        raise
    return catalog


def write_catalog(
    catalog: WorkspaceCatalog,
    path: Path | str | None = None,
    *,
    refresh_paths: bool = False,
    write_backup: bool = True,
) -> Path:
    """Atomically persist a catalog.

    Writes only under ``.behavior_suite/``; never rewrites scientific outputs.
    """

    target = Path(path) if path is not None else catalog_path_for_root(catalog.project.root_path)
    target = target if target.is_absolute() else target.resolve(strict=False)

    to_write = with_updated_path_status(catalog) if refresh_paths else catalog
    to_write = to_write.model_copy(
        update={
            "schema_version": SCHEMA_VERSION,
            "updated_utc": utc_now_iso(),
        }
    )
    # Re-validate uniqueness after refresh.
    parse_catalog_payload(catalog_to_jsonable(to_write))

    if write_backup and target.is_file():
        backup = target.with_suffix(target.suffix + CATALOG_BACKUP_SUFFIX)
        try:
            shutil.copy2(target, backup)
        except OSError as exc:
            raise WorkspaceCatalogError(
                f"Could not write catalog backup at {backup}: {exc}"
            ) from exc

    try:
        _atomic_write_text(target, serialize_catalog(to_write))
    except OSError as exc:
        raise WorkspaceCatalogError(f"Could not write catalog at {target}: {exc}") from exc
    return target


@dataclass(frozen=True)
class CatalogLoadResult:
    """Outcome of a catalog load attempt."""

    catalog: WorkspaceCatalog | None
    path: Path
    recovered: bool = False
    quarantine_path: Path | None = None
    error: str | None = None


def _quarantine_malformed(path: Path, reason: str) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{stamp}")
    try:
        os.replace(path, quarantine)
    except OSError as exc:
        raise WorkspaceCatalogError(
            f"Malformed catalog at {path} ({reason}); quarantine failed: {exc}"
        ) from exc
    return quarantine


def load_catalog(path: Path | str) -> WorkspaceCatalog:
    """Load and validate a catalog. Raises on missing/malformed/unsupported."""

    catalog_path = Path(path).expanduser().resolve(strict=False)
    if not catalog_path.is_file():
        raise WorkspaceCatalogError(f"Workspace catalog not found: {catalog_path}")
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceCatalogError(
            f"Could not read workspace catalog at {catalog_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise WorkspaceCatalogError(
            f"Workspace catalog root must be an object: {catalog_path}"
        )
    try:
        assert_supported_schema_version(raw.get("schema_version"))
        return parse_catalog_payload(raw)
    except WorkspaceCatalogVersionError:
        raise
    except (WorkspaceValidationError, WorkspaceDuplicateError) as exc:
        raise WorkspaceCatalogError(str(exc)) from exc


def load_catalog_safe(path: Path | str) -> CatalogLoadResult:
    """Read-only-safe loader: quarantine malformed catalogs instead of crashing callers.

    Unsupported future schema versions are **not** quarantined; they remain in place
    and surface ``error`` so operators can upgrade the app.
    """

    catalog_path = Path(path).expanduser().resolve(strict=False)
    if not catalog_path.is_file():
        return CatalogLoadResult(
            catalog=None,
            path=catalog_path,
            error=f"Workspace catalog not found: {catalog_path}",
        )
    try:
        catalog = load_catalog(catalog_path)
        return CatalogLoadResult(catalog=catalog, path=catalog_path)
    except WorkspaceCatalogVersionError as exc:
        return CatalogLoadResult(catalog=None, path=catalog_path, error=str(exc))
    except WorkspaceCatalogError as exc:
        quarantine: Path | None = None
        try:
            quarantine = _quarantine_malformed(catalog_path, str(exc))
        except WorkspaceCatalogError as quarantine_exc:
            return CatalogLoadResult(
                catalog=None,
                path=catalog_path,
                error=f"{exc}; {quarantine_exc}",
            )
        return CatalogLoadResult(
            catalog=None,
            path=catalog_path,
            recovered=True,
            quarantine_path=quarantine,
            error=str(exc),
        )


def detect_duplicate_session_root(
    catalog: WorkspaceCatalog,
    session_root: Path | str,
    *,
    ignore_session_id: str | None = None,
) -> str | None:
    """Return the existing session_id if ``session_root`` is already registered."""

    key = path_to_storage(session_root)
    for session in catalog.sessions:
        if ignore_session_id is not None and session.session_id == ignore_session_id:
            continue
        if path_to_storage(session.session_root) == key:
            return session.session_id
    return None


def replace_sessions(
    catalog: WorkspaceCatalog,
    sessions: tuple[SessionRecord, ...] | list[SessionRecord],
) -> WorkspaceCatalog:
    """Return catalog with replacement sessions and bumped ``updated_utc``."""

    updated = catalog.model_copy(
        update={
            "sessions": tuple(sessions),
            "updated_utc": utc_now_iso(),
        }
    )
    # Force uniqueness validation.
    return parse_catalog_payload(catalog_to_jsonable(updated))
