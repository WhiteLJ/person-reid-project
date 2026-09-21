"""Shared source-frame to OpenCV display-frame coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DisplayTransform:
    """Uniform, non-upscaling transform between source and display pixels."""

    source_width: int
    source_height: int
    display_width: int
    display_height: int
    scale: float

    @classmethod
    def from_frame(
        cls,
        frame: np.ndarray,
        max_display_width: int | None = 1280,
    ) -> "DisplayTransform":
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise ValueError("frame must be an image NumPy array")
        height, width = (int(frame.shape[0]), int(frame.shape[1]))
        return cls.from_size(width, height, max_display_width)

    @classmethod
    def from_size(
        cls,
        source_width: int,
        source_height: int,
        max_display_width: int | None = 1280,
    ) -> "DisplayTransform":
        source_width = int(source_width)
        source_height = int(source_height)
        if source_width <= 0 or source_height <= 0:
            raise ValueError("source dimensions must be positive")
        if max_display_width is None or int(max_display_width) <= 0:
            scale = 1.0
        else:
            max_width = int(max_display_width)
            scale = min(1.0, max_width / float(source_width))
        display_width = max(1, int(round(source_width * scale)))
        display_height = max(1, int(round(source_height * scale)))
        return cls(
            source_width=source_width,
            source_height=source_height,
            display_width=display_width,
            display_height=display_height,
            scale=scale,
        )

    @property
    def display_shape(self) -> tuple[int, int]:
        return self.display_height, self.display_width

    def source_to_display(self, frame: np.ndarray) -> np.ndarray:
        """Return a display-sized copy without changing the source frame."""

        self.validate_source_frame(frame)
        if (
            self.display_width == self.source_width
            and self.display_height == self.source_height
        ):
            return frame
        return cv2.resize(
            frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )

    def source_xyxy_to_display(
        self,
        bbox: Sequence[float],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self._validate_xyxy(bbox)
        return (
            self._clip_display_x(round(x1 * self.display_width / self.source_width)),
            self._clip_display_y(round(y1 * self.display_height / self.source_height)),
            self._clip_display_x(round(x2 * self.display_width / self.source_width)),
            self._clip_display_y(round(y2 * self.display_height / self.source_height)),
        )

    def display_xyxy_to_source(
        self,
        bbox: Sequence[float],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self._validate_xyxy(bbox)
        return (
            self._clip_source_x(round(x1 * self.source_width / self.display_width)),
            self._clip_source_y(round(y1 * self.source_height / self.display_height)),
            self._clip_source_x(round(x2 * self.source_width / self.display_width)),
            self._clip_source_y(round(y2 * self.source_height / self.display_height)),
        )

    def display_roi_xywh_to_source(
        self,
        roi: Sequence[float],
    ) -> tuple[int, int, int, int] | None:
        if len(roi) != 4:
            raise ValueError("ROI must contain exactly four values")
        x, y, width, height = (float(value) for value in roi)
        if not all(isfinite(value) for value in (x, y, width, height)):
            raise ValueError("ROI values must be finite")
        if width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = self.display_xyxy_to_source(
            (x, y, x + width, y + height)
        )
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    def _clip_display_x(self, value: int) -> int:
        return max(0, min(self.display_width, int(value)))

    def _clip_display_y(self, value: int) -> int:
        return max(0, min(self.display_height, int(value)))

    def _clip_source_x(self, value: int) -> int:
        return max(0, min(self.source_width, int(value)))

    def _clip_source_y(self, value: int) -> int:
        return max(0, min(self.source_height, int(value)))

    @staticmethod
    def _validate_frame_shape(
        frame: np.ndarray,
        expected_width: int,
        expected_height: int,
    ) -> None:
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim < 2
            or int(frame.shape[1]) != expected_width
            or int(frame.shape[0]) != expected_height
        ):
            raise ValueError("frame shape does not match DisplayTransform source size")

    def validate_source_frame(self, frame: np.ndarray) -> None:
        """Validate that an image uses this transform's source dimensions."""

        self._validate_frame_shape(frame, self.source_width, self.source_height)

    @staticmethod
    def _validate_xyxy(
        bbox: Sequence[float],
    ) -> tuple[float, float, float, float]:
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly four values")
        values = tuple(float(value) for value in bbox)
        if not all(isfinite(value) for value in values):
            raise ValueError("bbox values must be finite")
        x1, y1, x2, y2 = values
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
