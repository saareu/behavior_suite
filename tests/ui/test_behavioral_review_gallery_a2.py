"""Gallery A2 GUI: multi-select, working source combo, no silent draft creation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from tests.ui.test_behavioral_review_gui import _controller_with_package

from behavioral_review.contracts import (
    GALLERY_SOURCE_WORKING,
    STATUS_ACCEPTED,
)
from ui.controllers.behavioral_review_controller import (
    GALLERY_EDIT_REQUIRES_DRAFT_MESSAGE,
    VIEW_MODE_GALLERY,
    BehavioralReviewUiError,
)
from ui.pages.behavioral_review_page import BehavioralReviewPage
from ui.widgets.bout_gallery_view import BoutGalleryView

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def test_gallery_multi_select_is_explicit(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage(controller)
    page.refresh_from_state()
    controller.set_view_mode(VIEW_MODE_GALLERY)
    page.refresh_from_state()
    index = controller.state.gallery_index
    assert index is not None and index.n_cards >= 2
    view: BoutGalleryView = page.gallery_view
    first = index.cards[0].card_id
    second = index.cards[1].card_id
    # Plain click replaces selection.
    view._cards[first].clicked.emit(first, False)  # noqa: SLF001
    assert controller.state.gallery_selected_card_ids == (first,)
    assert view.selection_label.text() == "Selected: 1"
    # Ctrl-click adds.
    view._cards[second].clicked.emit(second, True)  # noqa: SLF001
    assert set(controller.state.gallery_selected_card_ids) == {first, second}
    assert view.selection_label.text() == "Selected: 2"
    # Explicit clear + source-kind change both drop selection.
    page._gallery_clear_selection()  # noqa: SLF001
    assert controller.state.gallery_selected_card_ids == ()
    view._cards[first].clicked.emit(first, False)  # noqa: SLF001
    assert controller.state.gallery_selected_card_ids == (first,)
    controller.set_reviewer_id("a2_select")
    assert controller.apply_edit_snapshot(
        controller.edit_worker(controller.begin_enter_editing())
    )
    assert controller.apply_edit_snapshot(
        controller.edit_worker(
            controller.begin_create_draft(review_root=tmp_path / "a2_select_drafts")
        )
    )
    controller.set_gallery_source_kind(GALLERY_SOURCE_WORKING)
    assert controller.state.gallery_selected_card_ids == ()
    del built


def test_gallery_review_requires_explicit_draft(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage(controller)
    page.refresh_from_state()
    controller.set_view_mode(VIEW_MODE_GALLERY)
    page.refresh_from_state()
    assert controller.state.draft_open is False
    assert page.gallery_source_combo.findData(GALLERY_SOURCE_WORKING) >= 0
    with pytest.raises(BehavioralReviewUiError, match="working draft"):
        controller.set_gallery_source_kind(GALLERY_SOURCE_WORKING)
    index = controller.state.gallery_index
    assert index is not None
    card_id = index.cards[0].card_id
    with pytest.raises(BehavioralReviewUiError) as exc:
        controller.begin_gallery_set_status(card_id, STATUS_ACCEPTED)
    assert GALLERY_EDIT_REQUIRES_DRAFT_MESSAGE[:24] in str(exc.value)
    assert controller.state.draft_open is False
    del built


def test_working_gallery_status_and_undo_via_controller(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    controller.set_reviewer_id("a2_ui")
    assert controller.apply_edit_snapshot(
        controller.edit_worker(controller.begin_enter_editing())
    )
    assert controller.apply_edit_snapshot(
        controller.edit_worker(
            controller.begin_create_draft(review_root=tmp_path / "a2_ui_drafts")
        )
    )
    controller.set_gallery_source_kind(GALLERY_SOURCE_WORKING)
    controller.set_view_mode(VIEW_MODE_GALLERY)
    index = controller.state.gallery_index
    assert index is not None and index.n_cards >= 1
    card = next(c for c in index.cards if c.source_s8_bout_id == "bout_dir_a")
    identity = controller.begin_gallery_set_status(card.card_id, STATUS_ACCEPTED)
    assert controller.apply_edit_snapshot(controller.edit_worker(identity))
    refreshed = controller.state.gallery_index
    assert refreshed is not None
    assert refreshed.card_by_id(card.card_id).status == STATUS_ACCEPTED
    # Undo restores prior decision via journal group.
    assert controller.apply_edit_snapshot(
        controller.edit_worker(controller.begin_undo())
    )
    assert (
        controller.state.gallery_index.card_by_id(card.card_id).status  # type: ignore[union-attr]
        != STATUS_ACCEPTED
    )
    del built
