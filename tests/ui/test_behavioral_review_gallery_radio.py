"""FIX-GALLERY-RADIO: Gallery/Timeline radio click path must drive view_mode."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)
from tests.ui.test_behavioral_review_gui import _controller_with_package

from ui.controllers.behavioral_review_controller import (
    VIEW_MODE_GALLERY,
    VIEW_MODE_TIMELINE,
)
from ui.pages.behavioral_review_page import BehavioralReviewPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _click_radio(radio) -> None:
    """Drive the real QRadioButton mouse-click path (not setChecked / controller)."""

    QTest.mouseClick(radio, Qt.MouseButton.LeftButton)
    _APPLICATION.processEvents()


def _assert_mode(page: BehavioralReviewPage, mode: str) -> None:
    assert page.controller.state.view_mode == mode
    if mode == VIEW_MODE_GALLERY:
        assert page.gallery_mode_radio.isChecked()
        assert not page.timeline_mode_radio.isChecked()
        assert page.view_stack.currentIndex() == 1
    else:
        assert page.timeline_mode_radio.isChecked()
        assert not page.gallery_mode_radio.isChecked()
        assert page.view_stack.currentIndex() == 0


def test_gallery_timeline_radio_clicks_drive_view_mode(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage(controller=controller)
    page.show()
    page.refresh_from_state()
    _APPLICATION.processEvents()

    # 1. Initial Timeline selection.
    _assert_mode(page, VIEW_MODE_TIMELINE)

    transitions: list[str] = []
    original_set_view_mode = controller.set_view_mode

    def _tracked_set_view_mode(mode: str) -> str:
        transitions.append(str(mode))
        return original_set_view_mode(mode)

    with patch.object(controller, "set_view_mode", side_effect=_tracked_set_view_mode):
        # 2. Click Gallery → controller enters Gallery.
        _click_radio(page.gallery_mode_radio)
        _assert_mode(page, VIEW_MODE_GALLERY)
        assert page.controller.state.gallery_index is not None
        assert page.controller.state.gallery_index.n_cards >= 1
        assert "exclusive bouts" in page.gallery_view.summary_label.text()
        assert len(page.gallery_view._cards) >= 1  # noqa: SLF001

        # 3. Click Timeline → controller returns to Timeline.
        _click_radio(page.timeline_mode_radio)
        _assert_mode(page, VIEW_MODE_TIMELINE)

        # 4. Repeat transitions without duplicate callbacks.
        _click_radio(page.gallery_mode_radio)
        _click_radio(page.timeline_mode_radio)
        _click_radio(page.gallery_mode_radio)
        _assert_mode(page, VIEW_MODE_GALLERY)

    assert transitions == [
        VIEW_MODE_GALLERY,
        VIEW_MODE_TIMELINE,
        VIEW_MODE_GALLERY,
        VIEW_MODE_TIMELINE,
        VIEW_MODE_GALLERY,
    ]

    # 5. Double-click gallery bout → Timeline radio and controller synchronize.
    card_id = page.controller.state.gallery_index.cards[0].card_id
    # Prefer the card widget double-click signal path when cards are realized.
    card_widget = page.gallery_view._cards.get(card_id)  # noqa: SLF001
    if card_widget is not None:
        QTest.mouseDClick(card_widget, Qt.MouseButton.LeftButton)
    else:
        page._gallery_open_in_timeline(card_id)  # noqa: SLF001
    _APPLICATION.processEvents()
    _assert_mode(page, VIEW_MODE_TIMELINE)
    assert page.controller.state.gallery_selected_card_id == card_id
    assert page.controller.state.draft_open is False
    assert page.controller.state.editing_active is False

    # 6. Source/session switch preserves a consistent mode selection.
    page.controller.state.view_mode = VIEW_MODE_GALLERY
    page.refresh_from_state()
    _assert_mode(page, VIEW_MODE_GALLERY)

    second = build_mini_review_package(tmp_path / "second_session")
    with patch(
        "results_packaging.runner.validate_s9_run_for_downstream",
        side_effect=lambda run_dir: {
            "s9_run_dir": Path(run_dir).resolve(),
            "status": "completed",
            "session_root": str(Path(run_dir).resolve().parent.parent),
            "source_s8_run_dir": Path(run_dir).resolve().parent.parent
            / "behavioral_arbitration"
            / second["s8_run_id"],
            "source_s8_run_id": second["s8_run_id"],
        },
    ), patch(
        "results_packaging.runner.is_s9_run_eligible_for_downstream",
        return_value=True,
    ), patch_s9_currently_eligible(source_s8_run_id=second["s8_run_id"]):
        identity = page.controller.set_session(second["session_root"])
        snap = page.controller.collect_discovery_snapshot(identity)
        assert page.controller.apply_discovery_snapshot(snap)
    page.refresh_from_state()
    _APPLICATION.processEvents()
    # Session reopen resets view mode to Timeline; radios must agree.
    _assert_mode(page, VIEW_MODE_TIMELINE)


def test_programmatic_view_mode_keeps_radios_in_sync(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage(controller=controller)
    page.show()
    page.refresh_from_state()
    _assert_mode(page, VIEW_MODE_TIMELINE)

    controller.set_view_mode(VIEW_MODE_GALLERY)
    page.refresh_from_state()
    _assert_mode(page, VIEW_MODE_GALLERY)

    controller.set_view_mode(VIEW_MODE_TIMELINE)
    page.refresh_from_state()
    _assert_mode(page, VIEW_MODE_TIMELINE)
