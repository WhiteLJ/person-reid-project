"""Small Atlas timing tool for detector, tracker, ReID, and business stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from src.ascend_detector import AscendPersonDetector
from src.ascend_reid import AscendReIDExtractor
from src.ascend_runtime import AscendRuntime
from src.config import load_config, parse_source
from src.gallery import TargetGallery
from src.gallery_recognition import GalleryRecognitionCoordinator
from src.reid import crop_person
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator
from src.ascend_tracker import AscendBotSortTracker
from src.video_source import VideoSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Atlas inference and CPU business stages")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", required=True, help="camera index or local video path")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--reid-samples-per-frame", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 1 or args.reid_samples_per_frame < 0:
        raise ValueError("frames must be positive and reid-samples-per-frame non-negative")
    config = load_config(args.config)
    runtime = AscendRuntime(config.ascend.device_id)
    source = VideoSource(parse_source(args.source))
    try:
        detector = AscendPersonDetector(
            config.ascend.yolo_model,
            runtime,
            image_size=config.model.image_size,
            confidence_threshold=config.model.conf_threshold,
            iou_threshold=config.model.iou_threshold,
            person_class_id=config.model.person_class_id,
        )
        tracker = AscendBotSortTracker(
            config.tracking.tracker,
            persist=config.tracking.persist,
        )
        reid = AscendReIDExtractor(
            config.reid,
            config.ascend.reid_model,
            config.ascend.reid_dynamic_batches,
            runtime,
        )
        target_manager = TargetManager()
        cache = ReIDFrameCache()
        recovery = TargetRecoveryCoordinator(
            target_manager,
            reid,  # type: ignore[arg-type]
            config.reid,
            config.reid_recovery,
            embedding_cache=cache,
            quality_config=config.reid_quality,
            person_class_id=config.model.person_class_id,
        )
        recognition = GalleryRecognitionCoordinator(
            target_manager,
            TargetGallery(),
            reid,  # type: ignore[arg-type]
            config.reid,
            config.gallery_recognition,
            config.reid_recovery,
            person_class_id=config.model.person_class_id,
            embedding_cache=cache,
            quality_config=config.reid_quality,
        )

        source.open()
        totals = np.zeros((6,), dtype=np.float64)
        processed = 0
        while processed < args.frames:
            frame = source.read()
            if frame is None:
                break
            total_start = perf_counter()
            start = perf_counter()
            detections = detector.detect(frame)
            totals[0] += perf_counter() - start
            start = perf_counter()
            tracks = tracker.update(detections, frame)
            totals[1] += perf_counter() - start

            crops = []
            for detection in detections[: args.reid_samples_per_frame]:
                crop = crop_person(
                    frame,
                    detection.bbox,
                    min_crop_width=config.reid.min_crop_width,
                    min_crop_height=config.reid.min_crop_height,
                )
                if crop is not None:
                    crops.append(crop)
            start = perf_counter()
            if crops:
                reid.extract_batch(crops)
            totals[2] += perf_counter() - start

            start = perf_counter()
            recovery.process_frame(frame, tracks, processed)
            totals[3] += perf_counter() - start
            start = perf_counter()
            recognition.process_frame(frame, tracks, processed)
            totals[4] += perf_counter() - start
            totals[5] += perf_counter() - total_start
            processed += 1

        if processed == 0:
            raise RuntimeError("benchmark source produced no frames")
        labels = (
            "YOLO NPU",
            "BoT-SORT CPU",
            "OSNet NPU",
            "Recovery ms",
            "Gallery ms",
            "Total ms",
        )
        for label, total in zip(labels, totals):
            print(f"{label}: {total * 1000.0 / processed:.3f}")
        print(f"frames: {processed}")
        print(f"FPS: {processed / totals[5]:.3f}")
        print(
            "Note: --reid-samples-per-frame measures an explicit ReID workload; "
            "the production pipeline still follows its interval/eligibility policy."
        )
        return 0
    finally:
        source.release()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
