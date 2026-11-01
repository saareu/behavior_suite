"""Behavior-grouped bout gallery grid (S10 Gallery A1/A2)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from behavioral_review.gallery import GalleryCard, GalleryGroup, GalleryIndex
from ui.widgets.video_frame_view import fit_image_target_rect

_SOURCE_COLORS = {
    "automated": "#455a64",
    "published": "#2e7d32",
    "working_draft": "#1565c0",
}

_STATUS_COLORS = {
    "unreviewed": "#607d8b",
    "accepted": "#2e7d32",
    "rejected": "#c62828",
    "voided": "#6d4c41",
}


class _PreviewPane(QWidget):
    """Tiny looping preview surface (BGR frames from the gallery scheduler)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        # Preferred (not Expanding) so QScrollArea + setWidgetResizable does not
        # collapse card heights to 0 and falsely mark every bout as visible.
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._frame = None
        else:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
                self._frame = None
            else:
                self._frame = np.ascontiguousarray(array, dtype=np.uint8)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#121212"))
        if self._frame is None:
            painter.setPen(QColor("#757575"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "…")
            return
        height, width = self._frame.shape[:2]
        target = fit_image_target_rect((width, height), (self.width(), self.height()))
        image = QImage(
            self._frame.data,
            width,
            height,
            int(self._frame.strides[0]),
            QImage.Format.Format_BGR888,
        )
        painter.drawImage(target, image)


class BoutGalleryCardWidget(QFrame):
    """One bout preview card with identity / label / status metadata."""

    clicked = Signal(str, bool)  # card_id, additive (Ctrl)
    open_requested = Signal(str)

    def __init__(self, card: GalleryCard, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.card_id = card.card_id
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected = False
        source_color = _SOURCE_COLORS.get(card.source_kind, "#455a64")
        status_color = _STATUS_COLORS.get(card.status, "#607d8b")
        self.setStyleSheet(
            f"BoutGalleryCardWidget {{ border: 2px solid {source_color}; "
            f"background: #263238; }}"
            "BoutGalleryCardWidget[selected='true'] { border-color: #ffeb3b; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.preview = _PreviewPane()
        layout.addWidget(self.preview, 1)
        title = QLabel(card.effective_label)
        title.setStyleSheet("font-weight: 600; color: #eceff1;")
        layout.addWidget(title)
        meta = QLabel(
            f"{card.bout_id}\n"
            f"focal {card.focal_id} → {card.partner_id} · "
            f"{card.bout_duration_frames} f ({card.bout_duration_sec:.3f}s)"
        )
        meta.setWordWrap(True)
        meta.setStyleSheet("color: #b0bec5; font-size: 11px;")
        layout.addWidget(meta)
        chips = QHBoxLayout()
        source_chip = QLabel(card.source_kind)
        source_chip.setStyleSheet(
            f"background: {source_color}; color: white; padding: 2px 6px;"
        )
        status_chip = QLabel(card.status)
        status_chip.setStyleSheet(
            f"background: {status_color}; color: white; padding: 2px 6px;"
        )
        chips.addWidget(source_chip)
        chips.addWidget(status_chip)
        if card.corrected:
            corr = QLabel("corrected")
            corr.setStyleSheet("background: #f9a825; color: #212121; padding: 2px 6px;")
            chips.addWidget(corr)
        chips.addStretch(1)
        layout.addLayout(chips)
        self.setToolTip(
            "Click to select · Ctrl+click to multi-select · "
            "Double-click to open in timeline/editor."
        )

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self.setProperty("selected", "true" if self._selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_preview_frame(self, frame: np.ndarray | None) -> None:
        self.preview.set_frame(frame)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            additive = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self.clicked.emit(self.card_id, additive)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_requested.emit(self.card_id)
        super().mouseDoubleClickEvent(event)


class BoutGalleryView(QWidget):
    """Scrollable label-grouped gallery grid with explicit multi-selection."""

    card_selected = Signal(str)
    selection_changed = Signal(object)  # tuple[str, ...]
    open_in_timeline_requested = Signal(str)
    visible_cards_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[str, BoutGalleryCardWidget] = {}
        self._selected_card_ids: tuple[str, ...] = ()
        self._columns = 3
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel("No gallery loaded.")
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)
        self.selection_label = QLabel("Selected: 0")
        self.selection_label.setStyleSheet("color: #90a4ae;")
        root.addWidget(self.selection_label)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.verticalScrollBar().valueChanged.connect(self._emit_visible)
        self.content = QWidget()
        # Minimum vertical policy keeps natural layout height so the viewport
        # scrolls instead of compressing every card into the visible rect.
        self.content.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self.grid = QVBoxLayout(self.content)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(10)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

    def set_index(
        self,
        index: GalleryIndex | None,
        *,
        selected_card_ids: tuple[str, ...] | None = None,
        selected_card_id: str | None = None,
    ) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards.clear()
        if selected_card_ids is not None:
            self._selected_card_ids = tuple(selected_card_ids)
        elif selected_card_id:
            self._selected_card_ids = (selected_card_id,)
        else:
            self._selected_card_ids = ()
        if index is None:
            self.content.setMinimumHeight(0)
            self.summary_label.setText("No gallery loaded.")
            self._update_selection_label()
            self.visible_cards_changed.emit(())
            return
        src = index.source.source_kind
        self.summary_label.setText(
            f"Gallery · {src} · {index.n_cards} exclusive bouts · "
            f"{index.n_groups} labels"
            + (
                f" · review {index.source.review_run_id}"
                if index.source.review_run_id
                else ""
            )
        )
        for group in index.groups:
            self._add_group(group)
        known = set(self._cards)
        self._selected_card_ids = tuple(
            cid for cid in self._selected_card_ids if cid in known
        )
        for card_id in self._selected_card_ids:
            self._cards[card_id].set_selected(True)
        self._update_selection_label()
        self._refresh_content_geometry()
        # Emit after the event loop applies scroll-area geometry. A synchronous
        # emit here can still see collapsed cards and mark every bout visible.
        QTimer.singleShot(0, self._emit_visible)

    def apply_preview_frame(self, card_id: str, frame: np.ndarray | None) -> None:
        widget = self._cards.get(card_id)
        if widget is None:
            return
        # Reject callbacks after Qt has destroyed the card (teardown / rebuild).
        if not isValid(widget):
            self._cards.pop(card_id, None)
            return
        try:
            widget.set_preview_frame(frame)
        except RuntimeError:
            # Underlying C++ object gone between isValid and paint path.
            self._cards.pop(card_id, None)

    def set_selected_cards(self, card_ids: tuple[str, ...] | None) -> None:
        for card_id in self._selected_card_ids:
            if card_id in self._cards:
                self._cards[card_id].set_selected(False)
        ordered = tuple(card_ids or ())
        self._selected_card_ids = tuple(cid for cid in ordered if cid in self._cards)
        for card_id in self._selected_card_ids:
            self._cards[card_id].set_selected(True)
        self._update_selection_label()

    def set_selected_card(self, card_id: str | None) -> None:
        if card_id:
            self.set_selected_cards((card_id,))
        else:
            self.set_selected_cards(())

    def selected_card_ids(self) -> tuple[str, ...]:
        return self._selected_card_ids

    def visible_card_ids(self) -> tuple[str, ...]:
        visible: list[str] = []
        viewport = self.scroll.viewport().rect()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return ()
        for card_id, widget in self._cards.items():
            if not isValid(widget):
                continue
            # Collapsed layout artifacts are not truly visible previews.
            if widget.width() < 32 or widget.height() < 64:
                continue
            top_left = widget.mapTo(self.scroll.viewport(), widget.rect().topLeft())
            bottom_right = widget.mapTo(
                self.scroll.viewport(), widget.rect().bottomRight()
            )
            card_rect = widget.rect()
            card_rect.moveTopLeft(top_left)
            card_rect.setBottomRight(bottom_right)
            if card_rect.intersects(viewport):
                visible.append(card_id)
        return tuple(visible)

    def _add_group(self, group: GalleryGroup) -> None:
        header = QLabel(
            f"{group.label}  (n={group.n_cards})  "
            f"✓{group.status_counts.get('accepted', 0)} "
            f"✗{group.status_counts.get('rejected', 0)} "
            f"?{group.status_counts.get('unreviewed', 0)}"
        )
        header.setStyleSheet(
            "font-weight: 700; color: #eceff1; padding: 4px 2px; "
            "border-bottom: 1px solid #546e7a;"
        )
        self.grid.addWidget(header)
        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        for i, card in enumerate(group.cards):
            widget = BoutGalleryCardWidget(card)
            widget.clicked.connect(self._on_card_clicked)
            widget.open_requested.connect(self.open_in_timeline_requested.emit)
            self._cards[card.card_id] = widget
            grid.addWidget(widget, i // self._columns, i % self._columns)
        self.grid.addWidget(wrap)

    def _on_card_clicked(self, card_id: str, additive: bool) -> None:
        if additive:
            current = list(self._selected_card_ids)
            if card_id in current:
                current = [c for c in current if c != card_id]
            else:
                current.append(card_id)
            self.set_selected_cards(tuple(current))
        else:
            self.set_selected_cards((card_id,))
        self.card_selected.emit(card_id)
        self.selection_changed.emit(self._selected_card_ids)

    def _update_selection_label(self) -> None:
        self.selection_label.setText(f"Selected: {len(self._selected_card_ids)}")

    def _emit_visible(self, *_args: object) -> None:
        self.visible_cards_changed.emit(self.visible_card_ids())

    def _refresh_content_geometry(self) -> None:
        """Keep natural content height under QScrollArea.setWidgetResizable.

        Without a minimum height, the scroll area repeatedly shrinks the content
        widget to the viewport, collapsing cards and making almost every bout
        appear visible to the preview scheduler.
        """

        layout = self.content.layout()
        if layout is None:
            return
        viewport_w = max(1, int(self.scroll.viewport().width()))
        self.content.setMinimumWidth(viewport_w)
        layout.activate()
        self.content.adjustSize()
        hint_h = max(
            int(layout.sizeHint().height()),
            int(layout.totalMinimumSize().height()),
            int(self.content.sizeHint().height()),
            int(self.content.height()),
        )
        # Fallback when Qt has not yet polished card widgets: reserve space from
        # preview minimums so the scroll area cannot collapse the grid.
        if hint_h < 64 and self._cards:
            rows = (len(self._cards) + self._columns - 1) // self._columns
            hint_h = max(hint_h, rows * 160 + max(1, len(self._cards)) * 8)
        self.content.setMinimumHeight(max(1, hint_h))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._cards:
            self._refresh_content_geometry()
        self._emit_visible()
