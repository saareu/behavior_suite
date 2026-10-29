"""Dense ethogram / parallel-lane timeline with a single authoritative playhead."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from behavioral_review.ethogram import EthogramLaneMeta
from behavioral_review.timeline import TimelineSegment

_LABEL_COLORS: dict[str, QColor] = {
    "approach": QColor("#4caf50"),
    "retract": QColor("#ff9800"),
    "engaged": QColor("#e91e63"),
    "nose_to_body": QColor("#9c27b0"),
    "mutually_disengaged": QColor("#03a9f4"),
    "undefined": QColor("#607d8b"),
    "1": QColor("#8bc34a"),
    "0": QColor("#37474f"),
    "contact": QColor("#8bc34a"),
    "side_by_side": QColor("#00bcd4"),
}

_LANE_HEIGHT = 22
_LANE_GAP = 4
_LEFT_GUTTER = 118


class EthogramTimelineView(QWidget):
    """Paint windowed ethogram segments; emit dense sample seeks on click/drag."""

    sample_seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._start = 0
        self._end = 0
        self._playhead = 0
        self._lanes: tuple[EthogramLaneMeta, ...] = ()
        self._segments: tuple[TimelineSegment, ...] = ()
        self._lod = "full"
        self.setMinimumHeight(140)
        self.setMouseTracking(True)

    def set_timeline(
        self,
        *,
        start_sample_idx: int,
        end_sample_idx: int,
        playhead_sample_idx: int,
        lanes: Sequence[EthogramLaneMeta],
        segments: Sequence[TimelineSegment],
        lod_level: str = "full",
    ) -> None:
        self._start = int(start_sample_idx)
        self._end = max(int(end_sample_idx), self._start)
        self._playhead = int(playhead_sample_idx)
        self._lanes = tuple(lanes)
        self._segments = tuple(segments)
        self._lod = str(lod_level)
        needed = max(
            140,
            24 + len(self._lanes) * (_LANE_HEIGHT + _LANE_GAP) + 8,
        )
        if self.minimumHeight() != needed:
            self.setMinimumHeight(needed)
        self.update()

    def clear(self) -> None:
        self._start = 0
        self._end = 0
        self._playhead = 0
        self._lanes = ()
        self._segments = ()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        if not self._lanes or self._end < self._start:
            painter.setPen(QColor("#9e9e9e"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No ethogram window loaded",
            )
            return

        span = max(1, self._end - self._start + 1)
        track_left = _LEFT_GUTTER
        track_width = max(1, self.width() - track_left - 8)
        y = 8
        for lane in self._lanes:
            painter.setPen(QColor("#cfd8dc"))
            painter.drawText(
                4,
                y,
                _LEFT_GUTTER - 8,
                _LANE_HEIGHT,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                lane.display_name,
            )
            lane_rect_y = y
            painter.fillRect(
                track_left,
                lane_rect_y,
                track_width,
                _LANE_HEIGHT,
                QColor("#263238"),
            )
            for seg in self._segments:
                if seg.lane_id != lane.lane_id:
                    continue
                x0 = track_left + int(
                    (seg.start_sample_idx - self._start) / span * track_width
                )
                x1 = track_left + int(
                    (seg.end_sample_idx - self._start + 1) / span * track_width
                )
                width = max(1, x1 - x0)
                color = _LABEL_COLORS.get(str(seg.value), QColor("#90a4ae"))
                if seg.lod_merged:
                    color = QColor(color.red(), color.green(), color.blue(), 180)
                painter.fillRect(x0, lane_rect_y + 2, width, _LANE_HEIGHT - 4, color)
            y += _LANE_HEIGHT + _LANE_GAP

        # Playhead (dense authority).
        if self._start <= self._playhead <= self._end:
            px = track_left + int(
                (self._playhead - self._start + 0.5) / span * track_width
            )
            painter.setPen(QPen(QColor("#ffeb3b"), 2))
            painter.drawLine(px, 4, px, self.height() - 4)

        painter.setPen(QColor("#78909c"))
        painter.drawText(
            4,
            self.height() - 18,
            self.width() - 8,
            16,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"samples [{self._start} … {self._end}]  playhead={self._playhead}  "
            f"lod={self._lod}",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            sample = self._sample_from_pos(event.position().toPoint())
            if sample is not None:
                self.sample_seek_requested.emit(sample)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            sample = self._sample_from_pos(event.position().toPoint())
            if sample is not None:
                self.sample_seek_requested.emit(sample)
        super().mouseMoveEvent(event)

    def _sample_from_pos(self, pos: QPoint) -> int | None:
        if self._end < self._start:
            return None
        track_left = _LEFT_GUTTER
        track_width = max(1, self.width() - track_left - 8)
        x = pos.x() - track_left
        if x < 0 or x > track_width:
            return None
        span = max(1, self._end - self._start + 1)
        frac = min(1.0, max(0.0, x / track_width))
        return int(self._start + frac * (span - 1))
