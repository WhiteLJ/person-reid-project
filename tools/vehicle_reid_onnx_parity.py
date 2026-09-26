"""Compare the PC Vehicle PyTorch model with its ONNX Runtime export."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.config import load_config
from src.vehicle_reid import VehicleReIDExtractor, _normalize_features


def _load_crops(paths: Sequence[str]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for value in paths:
        path = Path(value)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"vehicle image could not be read: {path}")
        crops.append(image)
    return crops


def _synthetic_crops(count: int) -> list[np.ndarray]:
    generator = np.random.default_rng(42)
    return [
        generator.integers(0, 256, (180, 320, 3), dtype=np.uint8)
        for _ in range(count)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vehicle PyTorch/ONNX Runtime embedding parity check"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("deploy/atlas/onnx/vehicle_sbs_r50_ibn.onnx"),
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="batch sizes to test; include 8 for the optional larger batch",
    )
    parser.add_argument(
        "--vehicle-image",
        nargs="*",
        default=(),
        help="optional BGR vehicle crops; synthetic crops are used otherwise",
    )
    parser.add_argument("--min-cosine", type=float, default=0.999)
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Vehicle parity requires PyTorch and onnxruntime"
        ) from exc

    if not args.onnx.is_file():
        raise FileNotFoundError(f"Vehicle ONNX file not found: {args.onnx}")
    if not args.batches or any(batch < 1 for batch in args.batches):
        raise ValueError("--batches must contain positive values")
    if args.min_cosine <= -1.0 or args.min_cosine > 1.0:
        raise ValueError("--min-cosine must be in (-1, 1]")

    config = load_config(args.config)
    extractor = VehicleReIDExtractor(config.vehicle_reid, device="cpu")
    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    if input_name != "images":
        raise RuntimeError(f"Vehicle ONNX input must be images, found {input_name}")

    max_batch = max(args.batches)
    crops = _load_crops(args.vehicle_image) if args.vehicle_image else []
    if not crops:
        crops = _synthetic_crops(max_batch)
    elif len(crops) < max_batch:
        crops = [crops[index % len(crops)] for index in range(max_batch)]

    all_cosines: list[float] = []
    print(f"onnx={args.onnx}")
    print(f"input={input_name} providers={session.get_providers()}")
    for batch_size in args.batches:
        batch = extractor._prepare_batch(crops[:batch_size])
        with torch.no_grad():
            torch_raw = extractor.model(batch)
        torch_embedding = _normalize_features(
            torch_raw.detach().cpu().numpy()
        )
        onnx_raw = session.run(None, {input_name: batch.numpy()})[0]
        onnx_embedding = _normalize_features(onnx_raw)
        if torch_embedding.shape != (batch_size, 2048):
            raise RuntimeError(
                f"PyTorch output must be ({batch_size},2048), "
                f"found {torch_embedding.shape}"
            )
        if onnx_embedding.shape != (batch_size, 2048):
            raise RuntimeError(
                f"ONNX output must be ({batch_size},2048), "
                f"found {onnx_embedding.shape}"
            )
        cosines = np.sum(torch_embedding * onnx_embedding, axis=1)
        all_cosines.extend(float(value) for value in cosines)
        print(
            f"batch={batch_size} shape={onnx_embedding.shape} "
            f"torch_norm={np.linalg.norm(torch_embedding, axis=1).mean():.6f} "
            f"onnx_norm={np.linalg.norm(onnx_embedding, axis=1).mean():.6f} "
            f"cosine_min={cosines.min():.6f} cosine_mean={cosines.mean():.6f}"
        )

    minimum = min(all_cosines)
    maximum = max(all_cosines)
    mean = float(np.mean(all_cosines))
    print(f"cosine_overall min={minimum:.6f} mean={mean:.6f} max={maximum:.6f}")
    if minimum < args.min_cosine:
        print(
            f"PARITY_WARN cosine_min={minimum:.6f} "
            f"below_min_cosine={args.min_cosine:.6f}"
        )
        return 1
    print(f"PARITY_OK cosine_min={minimum:.6f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
