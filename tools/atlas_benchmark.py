"""Stage and latency-distribution benchmark for the Atlas backend.

The benchmark deliberately measures the integrated frame path. It does not
replace production candidate selection with a smaller synthetic list. Use a
persisted Gallery database when measuring Gallery recognition load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from src.benchmark_utils import percentile_stats
from src.config import load_config, parse_source
from src.database import GalleryRepository
from src.gallery import TargetGallery
from src.gallery_recognition import GalleryRecognitionCoordinator
from src.gallery_service import GalleryPersistenceService
from src.reid import ReIDExtractor, crop_person
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator
from src.tracking_pipeline import TrackingPipeline
from src.video_source import VideoSource
from src.visualization import draw_tracks


STAGES = (
    "video_read",
    "yolo_preprocess",
    "yolo_h2d",
    "yolo_npu",
    "yolo_d2h",
    "yolo_decode_nms",
    "botsort",
    "osnet_preprocess",
    "osnet_npu",
    "recovery",
    "gallery_recognition",
    "render",
    "imshow_waitKey",
    "total",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Atlas YOLO/OSNet latency and frame-time jitter"
    )
    parser.add_argument("--config", default="config/config_atlas.yaml")
    parser.add_argument("--source", required=True, help="camera index or local video path")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--reid-samples-per-frame",
        type=int,
        default=0,
        help="optional explicit ReID workload; 0 measures production scheduling only",
    )
    parser.add_argument(
        "--frame-log",
        type=Path,
        help="optional JSONL output containing per-frame timing and OSNet batches",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="measure cv2.imshow/waitKey and show the benchmark window",
    )
    return parser


def _append(samples: dict[str, list[float]], name: str, value: float) -> None:
    samples[name].append(max(0.0, float(value)))


def _print_stats(samples: dict[str, list[float]]) -> None:
    print("stage                    count   average_ms   p50_ms   p95_ms   p99_ms   max_ms")
    for stage in STAGES:
        stats = percentile_stats(samples[stage])
        print(
            f"{stage:24s} {stats['count']:5d}   "
            f"{stats['average'] * 1000.0:10.3f} "
            f"{stats['p50'] * 1000.0:8.3f} "
            f"{stats['p95'] * 1000.0:8.3f} "
            f"{stats['p99'] * 1000.0:8.3f} "
            f"{stats['max'] * 1000.0:8.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 1 or args.reid_samples_per_frame < 0:
        raise ValueError("frames must be positive and reid-samples-per-frame non-negative")

    config = load_config(args.config)
    if config.inference.backend != "ascend":
        raise ValueError("atlas_benchmark requires inference.backend=ascend")

    runtime = None
    source = None
    log_stream = None
    window_name = "Atlas benchmark"
    try:
        from src.ascend_runtime import AscendRuntime

        runtime = AscendRuntime(config.ascend.device_id)
        source = VideoSource(parse_source(args.source))
        repository = GalleryRepository(config.database.path)
        gallery = TargetGallery()
        gallery_service = GalleryPersistenceService(
            gallery,
            repository,
            enrichment_config=config.gallery_enrichment,
        )
        gallery_service.load()

        tracking_pipeline = TrackingPipeline(
            model_config=config.model,
            runtime_config=config.runtime,
            tracking_config=config.tracking,
            inference_config=config.inference,
            ascend_config=config.ascend,
            ascend_runtime=runtime,
        )
        detector = tracking_pipeline.detector
        tracker = tracking_pipeline.tracker
        if detector is None or tracker is None:
            raise RuntimeError("Atlas tracking pipeline did not expose detector/tracker")
        reid_extractor = ReIDExtractor(
            config.reid,
            config.model.device,
            backend="ascend",
            ascend_config=config.ascend,
            ascend_runtime=runtime,
        )
        target_manager = TargetManager()
        cache = ReIDFrameCache()
        recovery = TargetRecoveryCoordinator(
            target_manager,
            reid_extractor,
            config.reid,
            config.reid_recovery,
            embedding_cache=cache,
            quality_config=config.reid_quality,
            person_class_id=config.model.person_class_id,
        )
        recognition = GalleryRecognitionCoordinator(
            target_manager,
            gallery,
            reid_extractor,
            config.reid,
            config.gallery_recognition,
            config.reid_recovery,
            person_class_id=config.model.person_class_id,
            embedding_cache=cache,
            quality_config=config.reid_quality,
        )

        if args.frame_log is not None:
            args.frame_log.parent.mkdir(parents=True, exist_ok=True)
            log_stream = args.frame_log.open("w", encoding="utf-8")

        if args.display:
            import cv2

        samples = {stage: [] for stage in STAGES}
        source.open()
        processed = 0
        unique_track_ids: set[int] = set()
        previous_track_ids: set[int] = set()
        track_created = 0
        track_ended = 0
        osnet_batch_sizes_total: list[int] = []
        gallery_candidate_counts: list[int] = []

        while processed < args.frames:
            frame_started = perf_counter()
            read_started = perf_counter()
            frame = source.read()
            read_seconds = perf_counter() - read_started
            if frame is None:
                break
            _append(samples, "video_read", read_seconds)
            frame_timings = {stage: 0.0 for stage in STAGES}
            frame_timings["video_read"] = read_seconds
            frame_index = processed

            detections = detector.detect(frame)
            detector_timing = getattr(detector, "last_timing", {})
            detector_stage_names = {
                "preprocess": "yolo_preprocess",
                "h2d": "yolo_h2d",
                "npu": "yolo_npu",
                "d2h": "yolo_d2h",
                "decode": "yolo_decode_nms",
            }
            for name, stage_name in detector_stage_names.items():
                value = detector_timing.get(name, 0.0)
                _append(samples, stage_name, value)
                frame_timings[stage_name] = float(value)

            tracker_started = perf_counter()
            tracks = tracker.update(detections, frame)
            tracker_seconds = perf_counter() - tracker_started
            _append(samples, "botsort", tracker_seconds)
            frame_timings["botsort"] = tracker_seconds
            current_track_ids = {track.track_id for track in tracks}
            track_created += len(current_track_ids - previous_track_ids)
            track_ended += len(previous_track_ids - current_track_ids)
            previous_track_ids = current_track_ids
            unique_track_ids.update(track.track_id for track in tracks)

            if args.reid_samples_per_frame:
                explicit_crops = []
                for detection in detections[: args.reid_samples_per_frame]:
                    crop = crop_person(
                        frame,
                        detection.bbox,
                        min_crop_width=config.reid.min_crop_width,
                        min_crop_height=config.reid.min_crop_height,
                    )
                    if crop is not None:
                        explicit_crops.append(crop)
                if explicit_crops:
                    reid_extractor.extract_batch(explicit_crops)

            recovery_started = perf_counter()
            recovery.process_frame(frame, tracks, frame_index)
            recovery_seconds = perf_counter() - recovery_started
            _append(samples, "recovery", recovery_seconds)
            frame_timings["recovery"] = recovery_seconds
            gallery_service.update_runtime_state(
                target_manager.targets.values(),
                frame_index,
            )
            gallery_service.enrich_reference_updates(
                recovery.drain_reference_updates()
            )

            recognition_started = perf_counter()
            recognition.process_frame(
                frame,
                tracks,
                frame_index,
                protected_track_ids=recovery.last_recovered_track_ids,
            )
            recognition_seconds = perf_counter() - recognition_started
            _append(samples, "gallery_recognition", recognition_seconds)
            frame_timings["gallery_recognition"] = recognition_seconds

            render_started = perf_counter()
            annotated = draw_tracks(
                frame,
                tracks,
                show_track_id=config.tracking.show_track_id,
                show_confidence=config.ui.show_confidence,
                class_name=tracking_pipeline.class_name,
                show_class_name=config.ui.show_class_name,
                selected_track_ids=target_manager.selected_track_ids,
                show_unselected_tracks=config.ui.show_unselected_tracks,
            )
            render_seconds = perf_counter() - render_started
            _append(samples, "render", render_seconds)
            frame_timings["render"] = render_seconds

            key = -1
            imshow_started = perf_counter()
            if args.display:
                import cv2

                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(config.ui.wait_key_ms) & 0xFF
            imshow_seconds = perf_counter() - imshow_started
            _append(samples, "imshow_waitKey", imshow_seconds)
            frame_timings["imshow_waitKey"] = imshow_seconds

            batch_events, timing_events = reid_extractor.drain_inference_diagnostics()
            for event in timing_events:
                preprocess_seconds = float(event.get("preprocess", 0.0))
                npu_seconds = float(event.get("npu", 0.0))
                _append(samples, "osnet_preprocess", preprocess_seconds)
                _append(samples, "osnet_npu", npu_seconds)
                frame_timings["osnet_preprocess"] += preprocess_seconds
                frame_timings["osnet_npu"] += npu_seconds
            frame_batch_sizes = [
                batch_size for event in batch_events for batch_size in event
            ]
            osnet_batch_sizes_total.extend(frame_batch_sizes)
            gallery_candidate_counts.append(recognition.last_candidate_count)

            total_seconds = perf_counter() - frame_started
            _append(samples, "total", total_seconds)
            frame_timings["total"] = total_seconds
            if log_stream is not None:
                log_stream.write(
                    json.dumps(
                        {
                            "frame": frame_index,
                            "track_count": len(tracks),
                            "gallery_candidate_count": recognition.last_candidate_count,
                            "osnet_batch_count": len(frame_batch_sizes),
                            "osnet_batch_sizes": frame_batch_sizes,
                            "timing_ms": {
                                stage: frame_timings[stage] * 1000.0
                                for stage in STAGES
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_stream.flush()

            processed += 1
            if key == ord("q"):
                break

        if processed == 0:
            raise RuntimeError("benchmark source produced no frames")
        _print_stats(samples)
        print(f"frames: {processed}")
        total_average = percentile_stats(samples["total"])["average"]
        average_fps = 1.0 / total_average if total_average > 0.0 else 0.0
        print(f"average_fps: {average_fps:.3f}")
        print(f"unique_track_ids: {len(unique_track_ids)}")
        print(f"track_created_observed: {track_created}")
        print(f"track_ended_observed: {track_ended}")
        print(f"target_lost: {recovery.target_manager.target_lost_count}")
        print(f"target_recovered: {recovery.target_manager.target_recovered_count}")
        print(f"recovery_attempted: {recovery.recovery_attempted_count}")
        print(f"recovery_pending: {recovery.recovery_pending_count}")
        print(f"recovery_accepted: {recovery.recovery_accepted_count}")
        print(f"quality_rejected: {recovery.quality_rejected_count + recognition.quality_rejected_count}")
        print(f"gallery_recognized: {recognition.recognized_count}")
        print(f"gallery_people: {len(gallery.all_people())}")
        print(
            "gallery_candidate_count_average/max: "
            f"{(sum(gallery_candidate_counts) / len(gallery_candidate_counts)):.2f}/"
            f"{max(gallery_candidate_counts, default=0)}"
        )
        print(f"osnet_batch_sizes_observed: {osnet_batch_sizes_total}")
        print(f"display: {'enabled' if args.display else 'disabled'}")
        if not args.display:
            print("imshow_waitKey is zero because --display was not supplied")
        if args.frame_log is not None:
            print(f"frame_log: {args.frame_log}")
        print(
            "Note: Gallery recognition uses the configured production candidate set; "
            "--reid-samples-per-frame adds an explicit diagnostic workload."
        )
        return 0
    finally:
        if log_stream is not None:
            log_stream.close()
        if source is not None:
            source.release()
        if args.display:
            try:
                import cv2

                cv2.destroyWindow(window_name)
            except Exception:
                pass
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
