"""S5 validation-frame bookmark persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from blob_features import (
    BlobFeaturesError,
    add_bookmark,
    bookmarks_path_for_session,
    empty_bookmark_store,
    load_bookmarks_or_empty,
    remove_bookmark,
    save_bookmarks,
)


def test_bookmark_round_trip(tmp_path: Path) -> None:
    session = tmp_path / "session"
    store = empty_bookmark_store(
        session_root=session,
        preprocess_dir=session / "preprocess",
        prepared_video=session / "preprocess" / "prepared_video.mp4",
    )
    store = add_bookmark(store, prepared_frame_idx=3, note="contact")
    store = add_bookmark(store, prepared_frame_idx=1, note="start")
    path = save_bookmarks(store)
    assert path == bookmarks_path_for_session(session)
    assert path.is_file()

    loaded = load_bookmarks_or_empty(
        session_root=session,
        preprocess_dir=session / "preprocess",
        prepared_video=session / "preprocess" / "prepared_video.mp4",
    )
    assert [item.prepared_frame_idx for item in loaded.bookmarks] == [1, 3]
    assert loaded.bookmarks[0].note == "start"

    loaded = remove_bookmark(loaded, 1)
    save_bookmarks(loaded)
    again = load_bookmarks_or_empty(
        session_root=session,
        preprocess_dir=session / "preprocess",
        prepared_video=session / "preprocess" / "prepared_video.mp4",
    )
    assert [item.prepared_frame_idx for item in again.bookmarks] == [3]


def test_duplicate_bookmark_replaces_note(tmp_path: Path) -> None:
    session = tmp_path / "session"
    store = empty_bookmark_store(
        session_root=session,
        preprocess_dir=session / "preprocess",
        prepared_video=session / "preprocess" / "prepared_video.mp4",
    )
    store = add_bookmark(store, prepared_frame_idx=2, note="a")
    store = add_bookmark(store, prepared_frame_idx=2, note="b")
    assert len(store.bookmarks) == 1
    assert store.bookmarks[0].note == "b"


def test_invalid_bookmark_index_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    store = empty_bookmark_store(
        session_root=session,
        preprocess_dir=session / "preprocess",
        prepared_video=session / "preprocess" / "prepared_video.mp4",
    )
    with pytest.raises(BlobFeaturesError):
        add_bookmark(store, prepared_frame_idx=-1)
