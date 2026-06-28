"""Aspect-ratio-preserving NumPy frame display widget and overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from preprocess.pre_crop import PreCropMode

_MANUAL_HANDLE_RADIUS_PX = 8.0
_DIRECTIONAL_PRE_CROP_MODES = {
    PreCropMode.VERTICAL_KEEP_LEFT.value,
    PreCropMode.VERTICAL_KEEP_RIGHT.value,
    PreCropMode.HORIZONTAL_KEEP_UPPER.value,
    PreCropMode.HORIZONTAL_KEEP_LOWER.value,
}


@dataclass(frozen=True, slots=True)
class PreCropOverlay:
    """Display-only visual state for one pre-crop selection.

    Coordinates are raw decode-frame pixel coordinates. Rectangles use the same
    half-open ``(x, y, width, height)`` convention as ``PreCropROI`` and
    ``PreCropConfig.manual_rectangle``.
    """

    mode: str = PreCropMode.NONE.value
    roi: tuple[int, int, int, int] | None = None
    boundary_px: int | None = None


def constrain_manual_pre_crop_rectangle(
    rectangle: tuple[int, int, int, int],
    frame_size_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Constrain a positive manual rectangle to raw-frame bounds.

    This helper is for visual manipulation only. Numeric configuration remains
    validated by the core ``resolve_pre_crop`` path, so invalid typed values are
    rejected rather than silently repaired.
    """

    x, y, width, height = rectangle
    frame_width, frame_height = frame_size_wh
    if width <= 0 or height <= 0:
        raise ValueError("Manual pre-crop rectangle must have positive size.")
    width = min(width, frame_width)
    height = min(height, frame_height)
    x = min(max(0, x), frame_width - width)
    y = min(max(0, y), frame_height - height)
    return x, y, width, height

def fit_image_target_rect(
    frame_size_wh: tuple[int, int],
    widget_size_wh: tuple[int, int],
) -> QRectF:
    """Return the display-only fitted rectangle for an unchanged image array."""

    frame_width, frame_height = frame_size_wh
    widget_width, widget_height = widget_size_wh
    available_width = max(1.0, float(widget_width - 8))
    available_height = max(1.0, float(widget_height - 8))
    scale = min(available_width / frame_width, available_height / frame_height)
    width = frame_width * scale
    height = frame_height * scale
    return QRectF(
        (widget_width - width) / 2,
        (widget_height - height) / 2,
        width,
        height,
    )


class VideoFrameView(QWidget):
    """Display one grayscale, BGR, or BGRA NumPy frame without retaining video."""

    pre_crop_geometry_changed = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._pre_crop_overlay = PreCropOverlay()
        self._drag_operation: str | None = None
        self._drag_anchor: tuple[int, int] | None = None
        self._drag_offset: tuple[int, int] | None = None
        self.setMinimumSize(320, 220)
        self.setMouseTracking(True)

    @property
    def frame_size_wh(self) -> tuple[int, int] | None:
        """Return the displayed source size in width/height order."""

        if self._frame is None:
            return None
        height, width = self._frame.shape[:2]
        return width, height

    def set_frame(self, frame: np.ndarray | None) -> None:
        """Copy one frame into widget-local display memory."""

        if frame is None:
            self._frame = None
        else:
            array = np.asarray(frame)
            if array.ndim not in {2, 3} or array.size == 0:
                raise ValueError("Displayed video frame must be a non-empty image array.")
            if array.ndim == 3 and array.shape[2] not in {3, 4}:
                raise ValueError("Displayed color frame must have three or four channels.")
            self._frame = np.ascontiguousarray(array, dtype=np.uint8)
        self.update()

    @property
    def pre_crop_overlay(self) -> PreCropOverlay:
        """Return the current display-only overlay state."""

        return self._pre_crop_overlay

    def set_pre_crop_overlay(self, overlay: PreCropOverlay | None) -> None:
        """Replace the display-only pre-crop overlay without emitting edits."""

        self._pre_crop_overlay = overlay or PreCropOverlay()
        self.update()

    def image_target_rect(self) -> QRectF:
        """Return the fitted on-widget rectangle used for image and overlays."""

        size = self.frame_size_wh
        if size is None:
            return QRectF()
        return fit_image_target_rect(size, (self.width(), self.height()))

    def widget_to_frame_point(self, point: QPointF) -> tuple[int, int] | None:
        """Map a widget point to a raw decode-frame pixel, excluding letterboxes."""

        fractional = self._widget_to_frame_fractional_point(point)
        size = self.frame_size_wh
        if fractional is None or size is None:
            return None
        frame_width, frame_height = size
        raw_x, raw_y = fractional
        return (
            max(0, min(frame_width - 1, int(raw_x))),
            max(0, min(frame_height - 1, int(raw_y))),
        )

    def frame_to_widget_rect(self, rectangle: tuple[int, int, int, int]) -> QRectF:
        """Map a raw-frame half-open rectangle to widget coordinates."""

        size = self.frame_size_wh
        target = self.image_target_rect()
        if size is None or target.isEmpty():
            return QRectF()
        frame_width, frame_height = size
        x, y, width, height = rectangle
        return QRectF(
            target.left() + (x / frame_width) * target.width(),
            target.top() + (y / frame_height) * target.height(),
            (width / frame_width) * target.width(),
            (height / frame_height) * target.height(),
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#181818"))
        if self._frame is None:
            painter.setPen(QColor("#b0b0b0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame loaded")
            return
        image = self._as_qimage(self._frame)
        target = self.image_target_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, image)
        self._draw_pre_crop_overlay(painter, target)
        painter.setPen(QColor("#555555"))
        painter.drawRect(target)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        position = event.position()
        frame_point = self.widget_to_frame_point(position)
        if frame_point is None:
            event.ignore()
            return
        mode = self._pre_crop_overlay.mode
        if mode in _DIRECTIONAL_PRE_CROP_MODES:
            self._drag_operation = "boundary"
            self._emit_boundary_for_position(mode, position)
            event.accept()
            return
        if mode != PreCropMode.MANUAL_RECTANGLE.value:
            event.ignore()
            return
        handle = self._manual_handle_at(position)
        if handle is not None:
            self._drag_operation = handle
            self._drag_anchor = self._opposite_manual_corner(handle)
            event.accept()
            return
        rectangle = self._pre_crop_overlay.roi
        if rectangle is not None and _point_is_inside_rectangle(frame_point, rectangle):
            x, y, _width, _height = rectangle
            self._drag_operation = "move"
            self._drag_offset = (frame_point[0] - x, frame_point[1] - y)
            event.accept()
            return
        self._drag_operation = "draw"
        self._drag_anchor = frame_point
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_operation is None:
            event.ignore()
            return
        position = event.position()
        frame_point = self.widget_to_frame_point(position)
        if frame_point is None:
            event.ignore()
            return
        mode = self._pre_crop_overlay.mode
        if self._drag_operation == "boundary" and mode in _DIRECTIONAL_PRE_CROP_MODES:
            self._emit_boundary_for_position(mode, position)
            event.accept()
            return
        if mode == PreCropMode.MANUAL_RECTANGLE.value:
            self._update_manual_drag(frame_point)
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_operation = None
            self._drag_anchor = None
            self._drag_offset = None
            event.accept()
            return
        event.ignore()

    @staticmethod
    def _as_qimage(frame: np.ndarray) -> QImage:
        height, width = frame.shape[:2]
        if frame.ndim == 2:
            image_format = QImage.Format.Format_Grayscale8
        elif frame.shape[2] == 3:
            image_format = QImage.Format.Format_BGR888
        else:
            image_format = QImage.Format.Format_BGRA8888
        return QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            image_format,
        )

    def _widget_to_frame_fractional_point(
        self,
        point: QPointF,
    ) -> tuple[float, float] | None:
        size = self.frame_size_wh
        target = self.image_target_rect()
        if size is None or target.isEmpty():
            return None
        if not (
            target.left() <= point.x() <= target.right()
            and target.top() <= point.y() <= target.bottom()
        ):
            return None
        frame_width, frame_height = size
        raw_x = ((point.x() - target.left()) / target.width()) * frame_width
        raw_y = ((point.y() - target.top()) / target.height()) * frame_height
        return (
            max(0.0, min(float(frame_width), raw_x)),
            max(0.0, min(float(frame_height), raw_y)),
        )

    def _draw_pre_crop_overlay(self, painter: QPainter, target: QRectF) -> None:
        overlay = self._pre_crop_overlay
        if overlay.mode == PreCropMode.NONE.value or overlay.roi is None:
            return
        retained = self.frame_to_widget_rect(overlay.roi)
        if retained.isEmpty():
            return
        painter.fillRect(
            QRectF(target.left(), target.top(), target.width(), retained.top() - target.top()),
            QColor(0, 0, 0, 105),
        )
        painter.fillRect(
            QRectF(target.left(), retained.bottom(), target.width(), target.bottom() - retained.bottom()),
            QColor(0, 0, 0, 105),
        )
        painter.fillRect(
            QRectF(target.left(), retained.top(), retained.left() - target.left(), retained.height()),
            QColor(0, 0, 0, 105),
        )
        painter.fillRect(
            QRectF(retained.right(), retained.top(), target.right() - retained.right(), retained.height()),
            QColor(0, 0, 0, 105),
        )
        painter.setPen(QColor("#30d158"))
        painter.drawRect(retained)
        if overlay.mode in _DIRECTIONAL_PRE_CROP_MODES and overlay.boundary_px is not None:
            self._draw_split_boundary(painter, target, overlay.mode, overlay.boundary_px)
        elif overlay.mode == PreCropMode.MANUAL_RECTANGLE.value:
            self._draw_manual_handles(painter, retained)

    def _draw_split_boundary(
        self,
        painter: QPainter,
        target: QRectF,
        mode: str,
        boundary_px: int,
    ) -> None:
        size = self.frame_size_wh
        if size is None:
            return
        frame_width, frame_height = size
        painter.setPen(QColor("#ffd60a"))
        if mode in {
            PreCropMode.VERTICAL_KEEP_LEFT.value,
            PreCropMode.VERTICAL_KEEP_RIGHT.value,
        }:
            x = target.left() + (boundary_px / frame_width) * target.width()
            painter.drawLine(QPointF(x, target.top()), QPointF(x, target.bottom()))
        else:
            y = target.top() + (boundary_px / frame_height) * target.height()
            painter.drawLine(QPointF(target.left(), y), QPointF(target.right(), y))

    def _draw_manual_handles(self, painter: QPainter, retained: QRectF) -> None:
        painter.fillRect(_handle_rect(retained.topLeft()), QColor("#30d158"))
        painter.fillRect(_handle_rect(retained.topRight()), QColor("#30d158"))
        painter.fillRect(_handle_rect(retained.bottomLeft()), QColor("#30d158"))
        painter.fillRect(_handle_rect(retained.bottomRight()), QColor("#30d158"))

    def _emit_boundary_for_position(self, mode: str, position: QPointF) -> None:
        fractional = self._widget_to_frame_fractional_point(position)
        size = self.frame_size_wh
        if fractional is None or size is None:
            return
        frame_width, frame_height = size
        raw_x, raw_y = fractional
        if mode == PreCropMode.VERTICAL_KEEP_LEFT.value:
            boundary = max(1, min(frame_width, int(round(raw_x))))
        elif mode == PreCropMode.VERTICAL_KEEP_RIGHT.value:
            boundary = max(0, min(frame_width - 1, int(round(raw_x))))
        elif mode == PreCropMode.HORIZONTAL_KEEP_UPPER.value:
            boundary = max(1, min(frame_height, int(round(raw_y))))
        else:
            boundary = max(0, min(frame_height - 1, int(round(raw_y))))
        if boundary != self._pre_crop_overlay.boundary_px:
            self.pre_crop_geometry_changed.emit(mode, boundary)

    def _manual_handle_at(self, position: QPointF) -> str | None:
        rectangle = self._pre_crop_overlay.roi
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
            if _distance_squared(position, handle_position) <= _MANUAL_HANDLE_RADIUS_PX**2:
                return name
        return None

    def _opposite_manual_corner(self, handle: str) -> tuple[int, int] | None:
        rectangle = self._pre_crop_overlay.roi
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

    def _update_manual_drag(self, frame_point: tuple[int, int]) -> None:
        size = self.frame_size_wh
        if size is None:
            return
        rectangle = self._pre_crop_overlay.roi
        if self._drag_operation == "move" and rectangle is not None:
            offset_x, offset_y = self._drag_offset or (0, 0)
            _old_x, _old_y, width, height = rectangle
            candidate = (frame_point[0] - offset_x, frame_point[1] - offset_y, width, height)
        else:
            anchor = self._drag_anchor
            if anchor is None:
                return
            left = min(anchor[0], frame_point[0])
            top = min(anchor[1], frame_point[1])
            right = max(anchor[0], frame_point[0])
            bottom = max(anchor[1], frame_point[1])
            candidate = (left, top, right - left, bottom - top)
        try:
            constrained = constrain_manual_pre_crop_rectangle(candidate, size)
        except ValueError:
            return
        if constrained != self._pre_crop_overlay.roi:
            self.pre_crop_geometry_changed.emit(
                PreCropMode.MANUAL_RECTANGLE.value,
                constrained,
            )


def _point_is_inside_rectangle(
    point: tuple[int, int],
    rectangle: tuple[int, int, int, int],
) -> bool:
    x, y, width, height = rectangle
    return x <= point[0] < x + width and y <= point[1] < y + height


def _handle_rect(point: QPointF) -> QRectF:
    return QRectF(
        point.x() - _MANUAL_HANDLE_RADIUS_PX / 2,
        point.y() - _MANUAL_HANDLE_RADIUS_PX / 2,
        _MANUAL_HANDLE_RADIUS_PX,
        _MANUAL_HANDLE_RADIUS_PX,
    )


def _distance_squared(first: QPointF, second: QPointF) -> float:
    return (first.x() - second.x()) ** 2 + (first.y() - second.y()) ** 2
