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
    parser.add_argument(
        "--exercise-recovery",
        action="store_true",
        help=(
            "create one temporary target, inject a synthetic loss, and measure "
            "the real Recovery ReID workload"
        ),
    )
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
        exercise_target_id: int | None = None
        exercise_selection_frame: int | None = None
        exercise_missing_frames = 0
        exercise_omitted_track_id: int | None = None
        recovery_workload_frames = 0
        recovery_candidate_count = 0
        recovery_quality_valid_count = 0
        recovery_reid_batch_count = 0
        recovery_reid_seconds = 0.0
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

            if args.exercise_recovery and exercise_target_id is None and tracks:
                selected = recovery.select_from_track(frame, tracks[0], processed)
                if selected is not None:
                    exercise_target_id = selected.target_id
                    exercise_selection_frame = processed
                    exercise_omitted_track_id = tracks[0].track_id

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
            recovery_tracks = tracks
            if (
                args.exercise_recovery
                and exercise_target_id is not None
                and exercise_selection_frame is not None
                and processed > exercise_selection_frame
            ):
                if exercise_missing_frames < config.reid_recovery.lost_grace_frames:
                    recovery_tracks = []
                    exercise_missing_frames += 1
                elif exercise_omitted_track_id is not None:
                    recovery_tracks = [
                        track
                        for track in tracks
                        if track.track_id != exercise_omitted_track_id
                    ]
            recovery.process_frame(frame, recovery_tracks, processed)
            totals[3] += perf_counter() - start
            recovery_stats = recovery.last_frame_recovery_stats
            if recovery_stats.recovery_due:
                recovery_workload_frames += 1
                recovery_candidate_count += recovery_stats.candidate_count
                recovery_quality_valid_count += recovery_stats.quality_valid_count
                recovery_reid_batch_count += recovery_stats.reid_batch_count
                recovery_reid_seconds += recovery_stats.reid_ms / 1000.0
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
        print(f"Recovery workload frames: {recovery_workload_frames}")
        print(f"Recovery candidates: {recovery_candidate_count}")
        print(f"Recovery quality-valid candidates: {recovery_quality_valid_count}")
        print(f"Recovery ReID batch executes: {recovery_reid_batch_count}")
        print(
            "Recovery ReID ms/frame: "
            f"{recovery_reid_seconds * 1000.0 / recovery_workload_frames:.3f}"
            if recovery_workload_frames
            else "Recovery ReID ms/frame: 0.000"
        )
        print(
            "Note: --reid-samples-per-frame measures an explicit ReID workload; "
            "the production pipeline still follows its interval/eligibility policy. "
            "Use --exercise-recovery to inject a benchmark-only target loss; "
            "it does not change the application pipeline."
        )
        return 0
    finally:
        source.release()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
