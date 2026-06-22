"""Aspect-ratio-preserving NumPy frame display widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget


class VideoFrameView(QWidget):
    """Display one grayscale, BGR, or BGRA NumPy frame without retaining video."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self.setMinimumSize(320, 220)

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

    def image_target_rect(self) -> QRectF:
        """Return the fitted on-widget rectangle used for image and overlays."""

        size = self.frame_size_wh
        if size is None:
            return QRectF()
        frame_width, frame_height = size
        available_width = max(1.0, float(self.width() - 8))
        available_height = max(1.0, float(self.height() - 8))
        scale = min(available_width / frame_width, available_height / frame_height)
        width = frame_width * scale
        height = frame_height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

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
        painter.setPen(QColor("#555555"))
        painter.drawRect(target)

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
