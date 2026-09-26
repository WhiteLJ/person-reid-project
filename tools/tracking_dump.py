"""Dump PC or Atlas detections/tracks for fixed-camera parity analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AppConfig, load_config, parse_source
from src.video_source import VideoSource


def _override_source(config: AppConfig, source: str) -> AppConfig:
    return replace(config, video=replace(config.video, source=parse_source(source)))


def _box_metrics(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "bbox": [x1, y1, x2, y2],
        "width": width,
        "height": height,
        "area": width * height,
        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
    }


def _detection_record(detection: Any) -> dict[str, Any]:
    record = _box_metrics(tuple(detection.bbox))
    record.update(
        {
            "class": int(detection.class_id),
            "confidence": float(detection.confidence),
        }
    )
    return record


def _track_record(track: Any, domain: str) -> dict[str, Any]:
    record = _box_metrics(tuple(track.bbox))
    record.update(
        {
            "track_id": int(track.track_id),
            "class": int(track.class_id),
            "confidence": float(track.confidence),
            "domain": domain,
        }
    )
    return record


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump detections and tracks for PC/Atlas bbox parity comparison"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def run(config_path: str, source_path: str, output_path: str, max_frames: int) -> int:
    config = _override_source(load_config(Path(config_path)), source_path)
    ascend_runtime = None
    if config.inference.backend == "torch":
        from src.pc_multiclass_tracking import MultiClassTrackingPipeline

        pipeline = MultiClassTrackingPipeline(config)
    elif config.inference.backend == "ascend":
        from src.ascend_multiclass_tracking import AscendMultiClassTrackingPipeline
        from src.ascend_runtime import AscendRuntime

        ascend_runtime = AscendRuntime(config.ascend.device_id)
        pipeline = AscendMultiClassTrackingPipeline(config, ascend_runtime)
    else:
        raise ValueError(f"unsupported inference backend: {config.inference.backend}")

    if max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = VideoSource(config.video.source)
    frame_count = 0
    try:
        source.open()
        with output.open("w", encoding="utf-8") as stream:
            while max_frames == 0 or frame_count < max_frames:
                frame = source.read()
                if frame is None:
                    break
                result = pipeline.process(frame)
                record = {
                    "frame_index": frame_count,
                    "source_shape": [int(value) for value in frame.shape[:2]],
                    "detections": [
                        _detection_record(detection)
                        for detection in pipeline.last_detections
                    ],
                    "tracks": {
                        "person": [
                            _track_record(track, "person")
                            for track in result.person_tracks
                        ],
                        "vehicle": [
                            _track_record(track, "vehicle")
                            for track in result.vehicle_tracks
                        ],
                    },
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                frame_count += 1
            stats = pipeline.stats()
            stream.write(
                json.dumps(
                    {
                        "type": "summary",
                        "frames": frame_count,
                        "yolo_inference_count": stats.yolo_inference_count,
                        "unique_person_tracks": stats.unique_person_tracks,
                        "unique_vehicle_tracks": stats.unique_vehicle_tracks,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        source.release()
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
        if ascend_runtime is not None:
            ascend_runtime.close()
    print(
        f"tracking_dump backend={config.inference.backend} frames={frame_count} "
        f"output={output}"
    )
    return 0


def main() -> int:
    args = build_argument_parser().parse_args()
    return run(args.config, args.source, args.output, args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
