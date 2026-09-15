"""Run a minimal YOLO OM + OSNet OM smoke test on an Atlas board."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.ascend_detector import AscendPersonDetector
from src.ascend_reid import AscendReIDExtractor
from src.ascend_runtime import AscendRuntime
from src.config import load_config
from src.reid import crop_person


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test Atlas YOLO and OSNet OM models")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--image", type=Path, required=True, help="BGR image used for YOLO inference")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="optional person crop in source-image xyxy coordinates")
    parser.add_argument("--yolo-model", type=Path)
    parser.add_argument("--reid-model", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    yolo_model = args.yolo_model or config.ascend.yolo_model
    reid_model = args.reid_model or config.ascend.reid_model
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"cannot read smoke-test image: {args.image}")

    runtime = AscendRuntime(config.ascend.device_id)
    try:
        detector = AscendPersonDetector(
            yolo_model,
            runtime,
            image_size=config.model.image_size,
            confidence_threshold=config.model.conf_threshold,
            iou_threshold=config.model.iou_threshold,
            person_class_id=config.model.person_class_id,
        )
        detections = detector.detect(frame)
        print(f"YOLO detections={len(detections)}")
        for detection in detections:
            print(
                f"bbox={detection.bbox} conf={detection.confidence:.6f} "
                f"class={detection.class_id}"
            )

        if args.crop is not None:
            bbox = tuple(float(value) for value in args.crop)
        elif detections:
            bbox = detections[0].bbox
        else:
            raise RuntimeError(
                "YOLO found no person; pass --crop X1 Y1 X2 Y2 to smoke-test OSNet"
            )
        crop = crop_person(
            frame,
            bbox,
            min_crop_width=config.reid.min_crop_width,
            min_crop_height=config.reid.min_crop_height,
        )
        if crop is None:
            raise RuntimeError("the smoke-test person crop is invalid or too small")

        extractor = AscendReIDExtractor(
            config.reid,
            reid_model,
            config.ascend.reid_dynamic_batches,
            runtime,
        )
        embedding = extractor.extract(crop)
        print(
            f"OSNet embedding shape={embedding.shape} dtype={embedding.dtype} "
            f"norm={np.linalg.norm(embedding):.6f}"
        )
        if embedding.shape != (512,) or embedding.dtype != np.float32:
            raise RuntimeError("OSNet smoke-test embedding contract failed")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

