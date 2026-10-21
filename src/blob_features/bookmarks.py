"""Lightweight GUI bookmarks for Subsystem 05 segmentation preview.

Bookmarks are diagnostic frame references associated with an eligible S1
preprocess source. They are not annotations, ground truth, or scientific
outputs of a completed extraction run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blob_features.contracts import BlobFeaturesError

BOOKMARKS_SCHEMA_VERSION = "s5_segmentation_bookmarks_v1"
BOOKMARKS_FILENAME = "segmentation_bookmarks.json"
_GUI_SUBDIR = "_gui"


@dataclass(frozen=True, slots=True)
class SegmentationBookmark:
    """One prepared-video frame bookmark for interactive preview."""

    prepared_frame_idx: int
    note: str = ""

    def validated(self) -> SegmentationBookmark:
        idx = int(self.prepared_frame_idx)
        if idx < 0:
            raise BlobFeaturesError(
                f"Bookmark prepared_frame_idx must be >= 0 (got {idx})."
            )
        note = str(self.note or "").strip()
        if len(note) > 500:
            raise BlobFeaturesError("Bookmark note must be at most 500 characters.")
        return SegmentationBookmark(prepared_frame_idx=idx, note=note)


@dataclass(frozen=True, slots=True)
class SegmentationBookmarkStore:
    """Persisted bookmark set tied to one S1 preprocess identity."""

    schema_version: str
    session_root: str
    preprocess_dir: str
    prepared_video: str
    bookmarks: tuple[SegmentationBookmark, ...]
    prepared_video_sha256: str = ""
    prepared_sync_sha256: str = ""

    def validated(self) -> SegmentationBookmarkStore:
        if self.schema_version != BOOKMARKS_SCHEMA_VERSION:
            raise BlobFeaturesError(
                f"Unsupported bookmarks schema_version: {self.schema_version}"
            )
        validated = tuple(item.validated() for item in self.bookmarks)
        # Stable unique order by frame index; last note wins on duplicates.
        by_idx: dict[int, SegmentationBookmark] = {}
        for item in validated:
            by_idx[item.prepared_frame_idx] = item
        ordered = tuple(
            by_idx[idx] for idx in sorted(by_idx)
        )
        return SegmentationBookmarkStore(
            schema_version=BOOKMARKS_SCHEMA_VERSION,
            session_root=str(self.session_root),
            preprocess_dir=str(self.preprocess_dir),
            prepared_video=str(self.prepared_video),
            bookmarks=ordered,
            prepared_video_sha256=str(self.prepared_video_sha256 or ""),
            prepared_sync_sha256=str(self.prepared_sync_sha256 or ""),
        )


def bookmarks_path_for_session(session_root: Path | str) -> Path:
    """Return the GUI bookmark file path under the session blob_features tree."""

    root = Path(session_root).expanduser().resolve(strict=False)
    return root / "blob_features" / _GUI_SUBDIR / BOOKMARKS_FILENAME


def empty_bookmark_store(
    *,
    session_root: Path | str,
    preprocess_dir: Path | str,
    prepared_video: Path | str,
    prepared_video_sha256: str = "",
    prepared_sync_sha256: str = "",
) -> SegmentationBookmarkStore:
    return SegmentationBookmarkStore(
        schema_version=BOOKMARKS_SCHEMA_VERSION,
        session_root=str(Path(session_root)),
        preprocess_dir=str(Path(preprocess_dir)),
        prepared_video=str(Path(prepared_video)),
        bookmarks=(),
        prepared_video_sha256=str(prepared_video_sha256 or ""),
        prepared_sync_sha256=str(prepared_sync_sha256 or ""),
    ).validated()


def load_bookmarks(path: Path | str) -> SegmentationBookmarkStore:
    """Load bookmarks JSON, or raise when the file is present but invalid."""

    bookmark_path = Path(path).expanduser().resolve(strict=False)
    if not bookmark_path.is_file():
        raise BlobFeaturesError(f"Bookmarks file not found: {bookmark_path}")
    try:
        raw = json.loads(bookmark_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlobFeaturesError(f"Could not load bookmarks: {exc}") from exc
    return bookmark_store_from_mapping(raw)


def load_bookmarks_or_empty(
    *,
    session_root: Path | str,
    preprocess_dir: Path | str,
    prepared_video: Path | str,
    prepared_sync: Path | str | None = None,
    prepared_video_sha256: str | None = None,
    prepared_sync_sha256: str | None = None,
) -> SegmentationBookmarkStore:
    """Load session bookmarks when present and identity matches; else empty.

    When the current S1 digests differ from the persisted store, bookmarks are
    not reused (source switching must not mix diagnostic frame lists).
    """

    video_path = Path(prepared_video)
    video_sha = prepared_video_sha256 or (
        _sha256(video_path) if video_path.is_file() else ""
    )
    sync_sha = ""
    if prepared_sync_sha256:
        sync_sha = prepared_sync_sha256
    elif prepared_sync is not None:
        sync_path = Path(prepared_sync)
        sync_sha = _sha256(sync_path) if sync_path.is_file() else ""

    path = bookmarks_path_for_session(session_root)
    if not path.is_file():
        return empty_bookmark_store(
            session_root=session_root,
            preprocess_dir=preprocess_dir,
            prepared_video=prepared_video,
            prepared_video_sha256=video_sha,
            prepared_sync_sha256=sync_sha,
        )
    store = load_bookmarks(path)
    if store.prepared_video_sha256 and store.prepared_video_sha256.lower() != video_sha.lower():
        return empty_bookmark_store(
            session_root=session_root,
            preprocess_dir=preprocess_dir,
            prepared_video=prepared_video,
            prepared_video_sha256=video_sha,
            prepared_sync_sha256=sync_sha,
        )
    if (
        store.prepared_sync_sha256
        and sync_sha
        and store.prepared_sync_sha256.lower() != sync_sha.lower()
    ):
        return empty_bookmark_store(
            session_root=session_root,
            preprocess_dir=preprocess_dir,
            prepared_video=prepared_video,
            prepared_video_sha256=video_sha,
            prepared_sync_sha256=sync_sha,
        )
    # Re-associate with the currently selected S1 paths / digests.
    return SegmentationBookmarkStore(
        schema_version=store.schema_version,
        session_root=str(Path(session_root)),
        preprocess_dir=str(Path(preprocess_dir)),
        prepared_video=str(Path(prepared_video)),
        bookmarks=store.bookmarks,
        prepared_video_sha256=video_sha,
        prepared_sync_sha256=sync_sha or store.prepared_sync_sha256,
    ).validated()


def save_bookmarks(store: SegmentationBookmarkStore, path: Path | str | None = None) -> Path:
    """Atomically write bookmarks JSON and return the written path."""

    validated = store.validated()
    target = (
        Path(path).expanduser().resolve(strict=False)
        if path is not None
        else bookmarks_path_for_session(validated.session_root)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": validated.schema_version,
        "session_root": validated.session_root,
        "preprocess_dir": validated.preprocess_dir,
        "prepared_video": validated.prepared_video,
        "prepared_video_sha256": validated.prepared_video_sha256,
        "prepared_sync_sha256": validated.prepared_sync_sha256,
        "bookmarks": [
            {"prepared_frame_idx": item.prepared_frame_idx, "note": item.note}
            for item in validated.bookmarks
        ],
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise BlobFeaturesError(f"Could not save bookmarks: {exc}") from exc
    return target


def bookmark_store_from_mapping(raw: Mapping[str, Any]) -> SegmentationBookmarkStore:
    if not isinstance(raw, Mapping):
        raise BlobFeaturesError("Bookmarks root must be a mapping.")
    bookmarks_raw = raw.get("bookmarks", [])
    if not isinstance(bookmarks_raw, Sequence) or isinstance(bookmarks_raw, (str, bytes)):
        raise BlobFeaturesError("Bookmarks 'bookmarks' must be a list.")
    items: list[SegmentationBookmark] = []
    for entry in bookmarks_raw:
        if not isinstance(entry, Mapping):
            raise BlobFeaturesError("Each bookmark must be a mapping.")
        items.append(
            SegmentationBookmark(
                prepared_frame_idx=int(entry["prepared_frame_idx"]),
                note=str(entry.get("note", "")),
            )
        )
    return SegmentationBookmarkStore(
        schema_version=str(raw.get("schema_version", BOOKMARKS_SCHEMA_VERSION)),
        session_root=str(raw.get("session_root", "")),
        preprocess_dir=str(raw.get("preprocess_dir", "")),
        prepared_video=str(raw.get("prepared_video", "")),
        bookmarks=tuple(items),
        prepared_video_sha256=str(raw.get("prepared_video_sha256", "")),
        prepared_sync_sha256=str(raw.get("prepared_sync_sha256", "")),
    ).validated()


def add_bookmark(
    store: SegmentationBookmarkStore,
    *,
    prepared_frame_idx: int,
    note: str = "",
) -> SegmentationBookmarkStore:
    item = SegmentationBookmark(prepared_frame_idx=prepared_frame_idx, note=note).validated()
    remaining = [
        existing
        for existing in store.bookmarks
        if existing.prepared_frame_idx != item.prepared_frame_idx
    ]
    remaining.append(item)
    return SegmentationBookmarkStore(
        schema_version=store.schema_version,
        session_root=store.session_root,
        preprocess_dir=store.preprocess_dir,
        prepared_video=store.prepared_video,
        bookmarks=tuple(remaining),
        prepared_video_sha256=store.prepared_video_sha256,
        prepared_sync_sha256=store.prepared_sync_sha256,
    ).validated()


def remove_bookmark(
    store: SegmentationBookmarkStore,
    prepared_frame_idx: int,
) -> SegmentationBookmarkStore:
    idx = int(prepared_frame_idx)
    filtered = tuple(
        item for item in store.bookmarks if item.prepared_frame_idx != idx
    )
    return SegmentationBookmarkStore(
        schema_version=store.schema_version,
        session_root=store.session_root,
        preprocess_dir=store.preprocess_dir,
        prepared_video=store.prepared_video,
        bookmarks=filtered,
        prepared_video_sha256=store.prepared_video_sha256,
        prepared_sync_sha256=store.prepared_sync_sha256,
    ).validated()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
