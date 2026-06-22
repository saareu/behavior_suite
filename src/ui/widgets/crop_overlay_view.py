"""Raw-frame crop review overlays and manual point input."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from preprocess.pre_crop import PreCropROI
from ui.widgets.video_frame_view import VideoFrameView

_POINT_LABELS = ("TL", "TR", "BR", "BL")


class CropOverlayView(VideoFrameView):
    """Overlay pre-crop/candidate geometry and collect raw-coordinate clicks."""

    raw_point_clicked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pre_crop_roi: PreCropROI | None = None
        self._candidate_quad: np.ndarray | None = None
        self._manual_points: tuple[tuple[float, float], ...] = ()
        self._manual_input_enabled = False

    def set_overlays(
        self,
        *,
        pre_crop_roi: PreCropROI | None,
        candidate_quad: np.ndarray | None,
        manual_points: tuple[tuple[float, float], ...],
        manual_input_enabled: bool,
    ) -> None:
        """Set raw-coordinate overlays without changing crop geometry."""

        self._pre_crop_roi = pre_crop_roi
        self._candidate_quad = (
            None
            if candidate_quad is None
            else np.asarray(candidate_quad, dtype=np.float64)
        )
        self._manual_points = tuple(manual_points)
        self._manual_input_enabled = manual_input_enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if manual_input_enabled
            else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            not self._manual_input_enabled
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mousePressEvent(event)
            return
        raw_point = self._view_to_raw(event.position())
        if raw_point is not None:
            self.raw_point_clicked.emit(raw_point.x(), raw_point.y())

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        if self.frame_size_wh is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._pre_crop_roi is not None:
            roi = self._pre_crop_roi
            top_left = self._raw_to_view((roi.x, roi.y))
            bottom_right = self._raw_to_view((roi.x + roi.width, roi.y + roi.height))
            painter.setPen(QPen(QColor("#42a5f5"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(
                min(top_left.x(), bottom_right.x()),
                min(top_left.y(), bottom_right.y()),
                abs(bottom_right.x() - top_left.x()),
                abs(bottom_right.y() - top_left.y()),
            )
        if self._candidate_quad is not None:
            self._draw_polyline(
                painter,
                [tuple(point) for point in self._candidate_quad],
                QColor("#66bb6a"),
                close=True,
            )
        if self._manual_points:
            self._draw_polyline(
                painter,
                list(self._manual_points),
                QColor("#ffb74d"),
                close=len(self._manual_points) == 4,
            )
            painter.setPen(QPen(QColor("#ffb74d"), 2))
            painter.setBrush(QColor("#ffb74d"))
            for index, point in enumerate(self._manual_points):
                mapped = self._raw_to_view(point)
                painter.drawEllipse(mapped, 5, 5)
                painter.drawText(
                    mapped + QPointF(7, -7),
                    _POINT_LABELS[index],
                )

    def _draw_polyline(
        self,
        painter: QPainter,
        points: list[tuple[float, float]],
        color: QColor,
        *,
        close: bool,
    ) -> None:
        mapped = [self._raw_to_view(point) for point in points]
        painter.setPen(QPen(color, 2))
        for first, second in zip(mapped, mapped[1:], strict=False):
            painter.drawLine(first, second)
        if close and len(mapped) > 2:
            painter.drawLine(mapped[-1], mapped[0])

    def _raw_to_view(self, point: tuple[float, float]) -> QPointF:
        size = self.frame_size_wh
        if size is None:
            return QPointF()
        width, height = size
        target = self.image_target_rect()
        return QPointF(
            target.left() + point[0] * target.width() / width,
            target.top() + point[1] * target.height() / height,
        )

    def _view_to_raw(self, point: QPointF) -> QPointF | None:
        size = self.frame_size_wh
        target = self.image_target_rect()
        if size is None or not target.contains(point):
            return None
        width, height = size
        raw_x = (point.x() - target.left()) * width / target.width()
        raw_y = (point.y() - target.top()) * height / target.height()
        return QPointF(
            min(max(raw_x, 0.0), np.nextafter(float(width), 0.0)),
            min(max(raw_y, 0.0), np.nextafter(float(height), 0.0)),
        )
