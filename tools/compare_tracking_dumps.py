"""Compare PC and Atlas detection geometry dumps frame by frame."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_dump(path: str | Path) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("type") == "summary":
                continue
            frames[int(record["frame_index"])] = record
    return frames


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(first: list[float], second: list[float]) -> float:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    return math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )


def _person_detections(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in record.get("detections", []) if int(item["class"]) == 0
    ]


def compare_records(
    pc: dict[int, dict[str, Any]],
    atlas: dict[int, dict[str, Any]],
    *,
    match_iou_threshold: float = 0.30,
    divergence_iou_threshold: float = 0.80,
    divergence_area_ratio: float = 1.30,
    divergence_center_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for frame_index in sorted(set(pc) | set(atlas)):
        pc_record = pc.get(frame_index, {})
        atlas_record = atlas.get(frame_index, {})
        pc_detections = _person_detections(pc_record)
        atlas_detections = _person_detections(atlas_record)
        pairs: list[tuple[float, int, int]] = []
        for pc_index, pc_detection in enumerate(pc_detections):
            for atlas_index, atlas_detection in enumerate(atlas_detections):
                pairs.append(
                    (
                        bbox_iou(pc_detection["bbox"], atlas_detection["bbox"]),
                        pc_index,
                        atlas_index,
                    )
                )
        pairs.sort(reverse=True)
        used_pc: set[int] = set()
        used_atlas: set[int] = set()
        matched: list[dict[str, Any]] = []
        for iou, pc_index, atlas_index in pairs:
            if iou < match_iou_threshold:
                break
            if pc_index in used_pc or atlas_index in used_atlas:
                continue
            used_pc.add(pc_index)
            used_atlas.add(atlas_index)
            pc_detection = pc_detections[pc_index]
            atlas_detection = atlas_detections[atlas_index]
            pc_area = float(pc_detection["area"])
            atlas_area = float(atlas_detection["area"])
            area_ratio = atlas_area / pc_area if pc_area > 0.0 else math.inf
            diagonal = max(
                math.hypot(float(pc_detection["width"]), float(pc_detection["height"])),
                1e-6,
            )
            center_ratio = _center_distance(
                pc_detection["bbox"], atlas_detection["bbox"]
            ) / diagonal
            reasons: list[str] = []
            if iou < divergence_iou_threshold:
                reasons.append("iou")
            if area_ratio > divergence_area_ratio:
                reasons.append("area_grew")
            elif area_ratio < 1.0 / divergence_area_ratio:
                reasons.append("area_shrank")
            if center_ratio > divergence_center_ratio:
                reasons.append("center_shift")
            item = {
                "frame_index": frame_index,
                "pc": pc_detection,
                "atlas": atlas_detection,
                "matched_bbox_iou": iou,
                "width_ratio": atlas_detection["width"] / max(float(pc_detection["width"]), 1e-6),
                "height_ratio": atlas_detection["height"] / max(float(pc_detection["height"]), 1e-6),
                "area_ratio": area_ratio,
                "center_distance": _center_distance(
                    pc_detection["bbox"], atlas_detection["bbox"]
                ),
                "center_distance_ratio": center_ratio,
                "confidence_difference": float(atlas_detection["confidence"])
                - float(pc_detection["confidence"]),
            }
            if reasons:
                item["diagnostic"] = "BBOX_GEOMETRY_DIVERGENCE"
                item["reasons"] = reasons
            matched.append(item)
        pc_tracks = pc_record.get("tracks", {}).get("person", [])
        atlas_tracks = atlas_record.get("tracks", {}).get("person", [])
        output.append(
            {
                "frame_index": frame_index,
                "pc_detection_count": len(pc_detections),
                "atlas_detection_count": len(atlas_detections),
                "pc_track_count": len(pc_tracks),
                "atlas_track_count": len(atlas_tracks),
                "matched": matched,
            }
        )
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PC/Atlas tracking dumps")
    parser.add_argument("--pc", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output")
    parser.add_argument("--match-iou-threshold", type=float, default=0.30)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    records = compare_records(
        _load_dump(args.pc),
        _load_dump(args.atlas),
        match_iou_threshold=args.match_iou_threshold,
    )
    divergent = sum(
        1
        for frame in records
        for match in frame["matched"]
        if match.get("diagnostic") == "BBOX_GEOMETRY_DIVERGENCE"
    )
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    print(
        f"compare_tracking_dumps frames={len(records)} "
        f"bbox_geometry_divergences={divergent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
