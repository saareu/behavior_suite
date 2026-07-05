"""Raw-frame crop review overlays and manual point input."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from preprocess.pre_crop import PreCropROI
from ui.widgets.video_frame_view import (
    VideoFrameView,
    constrain_manual_pre_crop_rectangle,
)

_POINT_LABELS = ("TL", "TR", "BR", "BL")


class CropOverlayView(VideoFrameView):
    """Overlay pre-crop/candidate geometry and collect raw-coordinate clicks."""

    raw_point_clicked = Signal(float, float)
    raw_rectangle_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pre_crop_roi: PreCropROI | None = None
        self._candidate_quad: np.ndarray | None = None
        self._manual_points: tuple[tuple[float, float], ...] = ()
        self._manual_rectangle_xywh: tuple[int, int, int, int] | None = None
        self._manual_input_enabled = False
        self._rectangle_input_enabled = False
        self._rectangle_drag_operation: str | None = None
        self._rectangle_drag_anchor: tuple[int, int] | None = None
        self._rectangle_drag_offset: tuple[int, int] | None = None

    def set_overlays(
        self,
        *,
        pre_crop_roi: PreCropROI | None,
        candidate_quad: np.ndarray | None,
        manual_points: tuple[tuple[float, float], ...],
        manual_rectangle_xywh: tuple[int, int, int, int] | None = None,
        manual_input_enabled: bool,
        rectangle_input_enabled: bool = False,
    ) -> None:
        """Set raw-coordinate overlays without changing crop geometry."""

        self._pre_crop_roi = pre_crop_roi
        self._candidate_quad = (
            None
            if candidate_quad is None
            else np.asarray(candidate_quad, dtype=np.float64)
        )
        self._manual_points = tuple(manual_points)
        self._manual_rectangle_xywh = manual_rectangle_xywh
        self._manual_input_enabled = manual_input_enabled
        self._rectangle_input_enabled = rectangle_input_enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if manual_input_enabled or rectangle_input_enabled
            else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._rectangle_input_enabled
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._handle_rectangle_press(event)
            return
        if (
            not self._manual_input_enabled
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mousePressEvent(event)
            return
        raw_point = self._view_to_raw(event.position())
        if raw_point is not None:
            self.raw_point_clicked.emit(raw_point.x(), raw_point.y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._rectangle_drag_operation is None:
            super().mouseMoveEvent(event)
            return
        boundary_point = self._view_to_raw_boundary(event.position())
        if boundary_point is not None:
            self._update_rectangle_drag(boundary_point)
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._rectangle_drag_operation is not None
        ):
            self._rectangle_drag_operation = None
            self._rectangle_drag_anchor = None
            self._rectangle_drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

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
        if self._manual_rectangle_xywh is not None:
            rect = self.frame_to_widget_rect(self._manual_rectangle_xywh)
            painter.setPen(QPen(QColor("#ffd60a"), 2))
            painter.drawRect(rect)
            self._draw_rectangle_handles(painter, rect)
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

    def _handle_rectangle_press(self, event: QMouseEvent) -> None:
        pixel_point = self.widget_to_frame_point(event.position())
        boundary_point = self._view_to_raw_boundary(event.position())
        if pixel_point is None or boundary_point is None:
            event.ignore()
            return
        handle = self._rectangle_handle_at(event.position())
        if handle is not None:
            self._rectangle_drag_operation = handle
            self._rectangle_drag_anchor = self._opposite_rectangle_corner(handle)
            event.accept()
            return
        rectangle = self._manual_rectangle_xywh
        if rectangle is not None and _point_is_inside_rectangle(pixel_point, rectangle):
            x, y, _width, _height = rectangle
            self._rectangle_drag_operation = "move"
            self._rectangle_drag_offset = (pixel_point[0] - x, pixel_point[1] - y)
            event.accept()
            return
        self._rectangle_drag_operation = "draw"
        self._rectangle_drag_anchor = boundary_point
        event.accept()

    def _update_rectangle_drag(self, boundary_point: tuple[int, int]) -> None:
        size = self.frame_size_wh
        if size is None:
            return
        rectangle = self._manual_rectangle_xywh
        if self._rectangle_drag_operation == "move" and rectangle is not None:
            offset_x, offset_y = self._rectangle_drag_offset or (0, 0)
            _x, _y, width, height = rectangle
            candidate = (
                boundary_point[0] - offset_x,
                boundary_point[1] - offset_y,
                width,
                height,
            )
        else:
            anchor = self._rectangle_drag_anchor
            if anchor is None:
                return
            left = min(anchor[0], boundary_point[0])
            top = min(anchor[1], boundary_point[1])
            right = max(anchor[0], boundary_point[0])
            bottom = max(anchor[1], boundary_point[1])
            candidate = (left, top, right - left, bottom - top)
        try:
            constrained = constrain_manual_pre_crop_rectangle(candidate, size)
        except ValueError:
            return
        if constrained != self._manual_rectangle_xywh:
            self.raw_rectangle_changed.emit(constrained)

    def _rectangle_handle_at(self, position: QPointF) -> str | None:
        rectangle = self._manual_rectangle_xywh
        if rectangle is None:
            return None
        retained = self.frame_to_widget_rect(rectangle)
        handles = {
            "resize_nw": retained.topLeft(),
            "resize_ne": retained.topRight(),
            "resize_sw": retained.bottomLeft(),
            "resize_se": retained.bottomRight(),
        }
        for name, handle_position in handles.items():
            if _distance_squared(position, handle_position) <= 64.0:
                return name
        return None

    def _opposite_rectangle_corner(self, handle: str) -> tuple[int, int] | None:
        rectangle = self._manual_rectangle_xywh
        if rectangle is None:
            return None
        x, y, width, height = rectangle
        if handle == "resize_nw":
            return x + width, y + height
        if handle == "resize_ne":
            return x, y + height
        if handle == "resize_sw":
            return x + width, y
        if handle == "resize_se":
            return x, y
        return None

    def _draw_rectangle_handles(self, painter: QPainter, retained: QRectF) -> None:
        painter.setBrush(QColor("#ffd60a"))
        painter.fillRect(_handle_rect(retained.topLeft()), QColor("#ffd60a"))
        painter.fillRect(_handle_rect(retained.topRight()), QColor("#ffd60a"))
        painter.fillRect(_handle_rect(retained.bottomLeft()), QColor("#ffd60a"))
        painter.fillRect(_handle_rect(retained.bottomRight()), QColor("#ffd60a"))

    def _view_to_raw_boundary(self, point: QPointF) -> tuple[int, int] | None:
        fractional = self._widget_to_frame_fractional_point(point)
        size = self.frame_size_wh
        if fractional is None or size is None:
            return None
        frame_width, frame_height = size
        return (
            max(0, min(frame_width, int(round(fractional[0])))),
            max(0, min(frame_height, int(round(fractional[1])))),
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


def _point_is_inside_rectangle(
    point: tuple[int, int],
    rectangle: tuple[int, int, int, int],
) -> bool:
    x, y, width, height = rectangle
    return x <= point[0] < x + width and y <= point[1] < y + height


def _handle_rect(point: QPointF) -> QRectF:
    return QRectF(point.x() - 4.0, point.y() - 4.0, 8.0, 8.0)


def _distance_squared(first: QPointF, second: QPointF) -> float:
    return (first.x() - second.x()) ** 2 + (first.y() - second.y()) ** 2
