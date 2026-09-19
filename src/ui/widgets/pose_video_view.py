"""Qt widget that displays one BGR frame with optional pose markers."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from ui.widgets.video_frame_view import fit_image_target_rect


class PoseVideoView(QWidget):
    """Aspect-preserving display for a prepared-video frame (already composited)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self.setMinimumSize(480, 360)

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._frame = None
        else:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
                raise ValueError("Pose video frame must be a non-empty BGR image.")
            self._frame = np.ascontiguousarray(array, dtype=np.uint8)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#181818"))
        if self._frame is None:
            painter.setPen(QColor("#b0b0b0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame loaded")
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
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, image)
        painter.setPen(QColor("#555555"))
        painter.drawRect(QRectF(target))
