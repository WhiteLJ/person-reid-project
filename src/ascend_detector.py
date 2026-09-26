"""Ascend OM YOLO preprocessing, decoding, and CPU postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .ascend_runtime import AscendModel, AscendRuntime
from .models import Detection


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometric transform used to map model coordinates to frame coordinates."""

    scale: float
    pad_x: float
    pad_y: float
    input_width: int
    input_height: int


def letterbox_bgr(
    frame: np.ndarray,
    image_size: int | tuple[int, int],
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize/pad a BGR frame and return an NCHW float32 model input."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (H, W, 3) in BGR order")
    if isinstance(image_size, int):
        input_height = input_width = image_size
    else:
        input_height, input_width = (int(image_size[0]), int(image_size[1]))
    if input_height < 1 or input_width < 1:
        raise ValueError("image_size must be positive")

    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("frame must not be empty")
    scale = min(input_width / width, input_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (input_width - resized_width) / 2.0
    pad_y = (input_height - resized_height) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    right = input_width - resized_width - left
    bottom = input_height - resized_height - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    tensor = (tensor / 255.0)[None, ...]
    return tensor, LetterboxTransform(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        input_width=input_width,
        input_height=input_height,
    )


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return indices kept by NumPy NMS for one already-selected class."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have matching lengths")
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_current = max(0.0, float(boxes[current, 2] - boxes[current, 0])) * max(
            0.0, float(boxes[current, 3] - boxes[current, 1])
        )
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        union = area_current + area_rest - intersection
        ious = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _prediction_matrix(output: np.ndarray) -> np.ndarray:
    """Normalize YOLO raw output to ``(num_predictions, channels)``."""

    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"unsupported YOLO output shape {array.shape}; expected [1,C,N] or [1,N,C]"
        )
    rows, columns = array.shape
    if columns < 5 and rows >= 5:
        return array.T
    if rows < 5 and columns >= 5:
        return array
    if rows in {6, 84, 85} and columns not in {6, 84, 85}:
        return array.T
    if columns in {6, 84, 85} and rows not in {6, 84, 85}:
        return array
    if rows <= 256 < columns:
        return array.T
    if columns <= 256 < rows:
        return array
    if rows <= 256 and columns > rows:
        return array.T
    if rows >= 5 and columns < 5:
        return array.T
    if columns < 5:
        raise ValueError(f"YOLO output has too few channels: {array.shape}")
    return array


def decode_yolo_output(
    output: np.ndarray,
    transform: LetterboxTransform,
    frame_shape: Sequence[int],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    person_class_id: int = 0,
) -> list[Detection]:
    """Decode YOLOv8/YOLO11 raw output and return person-only detections."""

    return decode_yolo_multiclass_output(
        output,
        transform,
        frame_shape,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        class_ids=(person_class_id,),
        target_class_id=person_class_id,
    )


def decode_yolo_multiclass_output(
    output: np.ndarray,
    transform: LetterboxTransform,
    frame_shape: Sequence[int],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    class_ids: Sequence[int],
    has_objectness: bool | None = None,
    target_class_id: int | None = None,
) -> list[Detection]:
    """Decode raw YOLO output for several classes with class-aware NMS.

    The project exports use the YOLOv8-style ``4 + class_scores`` layout.  The
    optional objectness argument keeps compatibility with older YOLO exports
    using ``4 + objectness + class_scores``.  When omitted, the historical
    project heuristic is retained for the common one-class/COCO-80 layouts.
    """

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    allowed_class_ids = tuple(sorted({int(class_id) for class_id in class_ids}))
    if not allowed_class_ids or any(class_id < 0 for class_id in allowed_class_ids):
        raise ValueError("class_ids must contain at least one non-negative class ID")
    predictions = _prediction_matrix(output)
    channels = predictions.shape[1]
    if channels < 5:
        raise ValueError(f"YOLO output has too few channels: {channels}")

    if has_objectness is None:
        # Preserve support for the legacy single-class and COCO-80 layouts.
        has_objectness = channels in {6, 85}
    class_start = 5 if has_objectness else 4
    class_count = channels - class_start
    if max(allowed_class_ids) >= class_count:
        return []
    boxes_xywh = predictions[:, :4]
    if has_objectness:
        class_scores = predictions[:, class_start:]
        effective_scores = predictions[:, 4:5] * class_scores
    else:
        effective_scores = predictions[:, class_start:]
    if target_class_id is not None:
        if target_class_id not in allowed_class_ids:
            raise ValueError("target_class_id must be included in class_ids")
        if target_class_id >= effective_scores.shape[1]:
            return []
        best_classes = np.full(
            (len(predictions),), target_class_id, dtype=np.int64
        )
        best_scores = effective_scores[:, target_class_id]
    else:
        best_indices = np.argmax(effective_scores, axis=1)
        best_scores = effective_scores[np.arange(len(predictions)), best_indices]
        best_classes = best_indices.astype(np.int64, copy=False)
    allowed = np.isin(best_classes, np.asarray(allowed_class_ids, dtype=np.int64))
    valid = (
        np.isfinite(predictions).all(axis=1)
        & np.isfinite(best_scores)
        & allowed
        & (best_scores >= confidence_threshold)
    )
    if not np.any(valid):
        return []

    boxes_xywh = boxes_xywh[valid]
    scores = best_scores[valid].astype(np.float32, copy=False)
    detected_classes = best_classes[valid]
    boxes = np.empty_like(boxes_xywh, dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_x) / transform.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_y) / transform.scale

    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(frame_width))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(frame_height))
    valid_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not np.any(valid_area):
        return []
    boxes = boxes[valid_area]
    scores = scores[valid_area]
    detected_classes = detected_classes[valid_area]
    detections: list[Detection] = []
    for class_id in allowed_class_ids:
        class_indices = np.flatnonzero(detected_classes == class_id)
        if class_indices.size == 0:
            continue
        kept = nms_xyxy(boxes[class_indices], scores[class_indices], iou_threshold)
        detections.extend(
            Detection(
                bbox=tuple(float(value) for value in boxes[class_indices[index]]),
                confidence=float(scores[class_indices[index]]),
                class_id=class_id,
            )
            for index in kept
        )
    return detections


class AscendMultiClassDetector:
    """Run one YOLO OM inference and decode the configured class set."""

    def __init__(
        self,
        model_path: str | Path,
        runtime: AscendRuntime,
        *,
        image_size: int,
        confidence_threshold: float,
        iou_threshold: float,
        class_ids: Sequence[int],
        model: AscendModel | Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.runtime = runtime
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_ids = tuple(sorted({int(class_id) for class_id in class_ids}))
        if not self.class_ids or any(class_id < 0 for class_id in self.class_ids):
            raise ValueError("class_ids must contain at least one non-negative class ID")
        self.model = model if model is not None else runtime.load_model(self.model_path)
        self.inference_count = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, transform = letterbox_bgr(frame, self.image_size)
        outputs = self.runtime.execute(self.model, [tensor])
        self.inference_count += 1
        if not outputs:
            raise ValueError("YOLO OM returned no outputs")
        output = outputs[0]
        output_shape = getattr(self.model, "output_shapes", ())
        if np.asarray(output).ndim == 1 and output_shape:
            shape = output_shape[0]
            if shape is not None and all(value > 0 for value in shape):
                if int(np.prod(shape)) == np.asarray(output).size:
                    output = np.asarray(output).reshape(shape)
        return decode_yolo_multiclass_output(
            output,
            transform,
            frame.shape,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            class_ids=self.class_ids,
        )


class AscendPersonDetector:
    """Run a project-exported YOLO OM and decode person detections on CPU."""

    def __init__(
        self,
        model_path: str | Path,
        runtime: AscendRuntime,
        *,
        image_size: int,
        confidence_threshold: float,
        iou_threshold: float,
        person_class_id: int = 0,
        model: AscendModel | Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.runtime = runtime
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self.model = (
            model if model is not None else runtime.load_model(self.model_path)
        )
        self.inference_count = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, transform = letterbox_bgr(frame, self.image_size)
        outputs = self.runtime.execute(self.model, [tensor])
        self.inference_count += 1
        if not outputs:
            raise ValueError("YOLO OM returned no outputs")
        output = outputs[0]
        output_shape = getattr(self.model, "output_shapes", ())
        if np.asarray(output).ndim == 1 and output_shape:
            shape = output_shape[0]
            if shape is not None and all(value > 0 for value in shape):
                if int(np.prod(shape)) == np.asarray(output).size:
                    output = np.asarray(output).reshape(shape)
        return decode_yolo_output(
            output,
            transform,
            frame.shape,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            person_class_id=self.person_class_id,
        )
