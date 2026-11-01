"""Gallery A1 lifecycle: visibility bounds, ordered teardown, stale callback reject."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow
from tests.ui.test_behavioral_review_gui import _controller_with_package

from behavioral_review.contracts import (
    GALLERY_CARD_SCHEMA_VERSION,
    GALLERY_INDEX_SCHEMA_VERSION,
    GALLERY_SOURCE_AUTOMATED,
    STATUS_UNREVIEWED,
)
from behavioral_review.gallery import (
    GalleryCard,
    GalleryGroup,
    GalleryIndex,
    GalleryPreviewPlan,
    GalleryQuery,
    GallerySourceRef,
)
from ui.controllers.behavioral_review_controller import (
    VIEW_MODE_GALLERY,
    VIEW_MODE_TIMELINE,
)
from ui.pages.behavioral_review_page import BehavioralReviewPage
from ui.widgets.bout_gallery_view import BoutGalleryView

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _synthetic_index(n_cards: int = 48) -> GalleryIndex:
    plan = GalleryPreviewPlan(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        clip_frame_idx_start=0,
        clip_frame_idx_end=2,
        clip_frame_count=3,
        pre_context_sec=0.0,
        post_context_sec=0.0,
        source_fps=30.0,
        source_n_frames=100,
        first_visible_prepared_frame_idx=0,
        last_visible_prepared_frame_idx=2,
    )
    cards: list[GalleryCard] = []
    for i in range(n_cards):
        cards.append(
            GalleryCard(
                schema_version=GALLERY_CARD_SCHEMA_VERSION,
                card_id=f"automated:bout_{i:04d}",
                bout_id=f"bout_{i:04d}",
                source_kind=GALLERY_SOURCE_AUTOMATED,
                channel="exclusive",
                identity_grain="directional",
                automated_label="approach",
                effective_label="approach",
                corrected=False,
                focal_id=0,
                partner_id=1,
                start_sample_idx=i * 10,
                end_sample_idx=i * 10 + 5,
                bout_duration_frames=6,
                bout_duration_sec=6 / 120.0,
                prepared_frame_idx_start=0,
                prepared_frame_idx_end=2,
                status=STATUS_UNREVIEWED,
                void_reason=None,
                source_s8_bout_id=f"s8_{i:04d}",
                package_bout_id=f"bout_{i:04d}",
                reviewed_bout_id=None,
                preview_plan=plan,
            )
        )
    group = GalleryGroup(
        label="approach",
        cards=tuple(cards),
        n_cards=len(cards),
        status_counts={STATUS_UNREVIEWED: len(cards)},
    )
    return GalleryIndex(
        schema_version=GALLERY_INDEX_SCHEMA_VERSION,
        source=GallerySourceRef(
            source_kind=GALLERY_SOURCE_AUTOMATED,
            package_run_id="pkg_test",
        ),
        cards=tuple(cards),
        groups=(group,),
        query=GalleryQuery(),
        n_cards=len(cards),
        n_groups=1,
        available_labels=("approach",),
        available_focal_ids=(0,),
        status_counts={STATUS_UNREVIEWED: len(cards)},
    )


def test_gallery_visible_cards_bounded_after_layout() -> None:
    """Regression: Expanding + QScrollArea must not mark every card visible."""

    window = QMainWindow()
    view = BoutGalleryView()
    window.setCentralWidget(view)
    window.resize(720, 480)
    window.show()
    _APPLICATION.processEvents()

    index = _synthetic_index(48)
    view.set_index(index)
    _APPLICATION.processEvents()
    view.content.adjustSize()
    _APPLICATION.processEvents()

    n_cards = len(view._cards)  # noqa: SLF001
    assert n_cards == 48
    # Natural content height must exceed the viewport (scrollable, not collapsed).
    assert view.content.height() > view.scroll.viewport().height()
    visible = view.visible_card_ids()
    assert 1 <= len(visible) < n_cards
    for card_id in visible:
        widget = view._cards[card_id]  # noqa: SLF001
        assert widget.width() > 0
        assert widget.height() > 0

    window.close()
    _APPLICATION.processEvents()


def test_gallery_shutdown_stops_timer_and_rejects_late_frames(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage(controller=controller)
    page.show()
    page.resize(800, 600)
    controller.set_view_mode(VIEW_MODE_GALLERY)
    page.refresh_from_state()
    _APPLICATION.processEvents()

    assert controller.state.gallery_preview_scheduler is not None
    page._start_gallery_previews()  # noqa: SLF001
    assert page._gallery_preview_timer.isActive()  # noqa: SLF001

    page.shutdown_gallery_surface()
    assert not page._gallery_preview_timer.isActive()  # noqa: SLF001
    assert controller.state.gallery_preview_scheduler is None

    # Late tick / frame application must be no-ops after shutdown.
    page._on_gallery_preview_tick()  # noqa: SLF001
    page.gallery_view.apply_preview_frame("missing-card", None)
    assert controller.state.gallery_preview_scheduler is None
    assert not page._gallery_preview_timer.isActive()  # noqa: SLF001

    page.close()
    _APPLICATION.processEvents()


def test_gallery_mode_churn_then_ordered_shutdown(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage(controller=controller)
    page.show()
    page.resize(800, 600)
    page.refresh_from_state()
    _APPLICATION.processEvents()

    for _ in range(3):
        controller.set_view_mode(VIEW_MODE_GALLERY)
        page.refresh_from_state()
        _APPLICATION.processEvents()
        page._on_gallery_preview_tick()  # noqa: SLF001
        controller.set_view_mode(VIEW_MODE_TIMELINE)
        page.refresh_from_state()
        _APPLICATION.processEvents()

    page.shutdown_gallery_surface()
    controller.close_session(release_lock=True)
    page.close()
    _APPLICATION.processEvents()
    assert controller.state.gallery_preview_scheduler is None
    assert controller.state.editing_session is None
