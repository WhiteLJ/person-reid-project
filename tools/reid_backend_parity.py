"""Compare Torch, ONNX Runtime, and Atlas OM OSNet feature spaces.

The ``export`` subcommand runs on the PC and stores the actual BGR crops plus
Torch/ONNX embeddings in a non-pickle NPZ file. ``check-om`` runs on Atlas and
compares OM output against those same stored samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def save_reference(
    path: str | Path,
    names: Sequence[str],
    crops: Sequence[np.ndarray],
    torch_embeddings: np.ndarray,
    onnx_embeddings: np.ndarray,
) -> None:
    """Save parity inputs and outputs without object arrays or pickle."""

    if len(names) != len(crops):
        raise ValueError("names and crops must have the same length")
    torch_values = np.asarray(torch_embeddings, dtype=np.float32)
    onnx_values = np.asarray(onnx_embeddings, dtype=np.float32)
    count = len(crops)
    if torch_values.shape != (count, 512) or onnx_values.shape != (count, 512):
        raise ValueError("Torch and ONNX embeddings must both have shape (N, 512)")
    for name, values in (("Torch", torch_values), ("ONNX", onnx_values)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} embeddings contain non-finite values")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"{name} embeddings must be L2 normalized")
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray([1], dtype=np.int32),
        "sample_count": np.asarray([count], dtype=np.int32),
        "sample_names": np.asarray(tuple(str(name) for name in names), dtype="U256"),
        "torch_embeddings": torch_values,
        "onnx_embeddings": onnx_values,
    }
    for index, crop in enumerate(crops):
        value = np.asarray(crop)
        if value.ndim != 3 or value.shape[2] != 3 or value.size == 0:
            raise ValueError(f"crop {index} must have shape (H, W, 3)")
        payload[f"crop_{index:04d}"] = np.ascontiguousarray(value, dtype=np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def load_reference(path: str | Path) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Load a parity reference using ``allow_pickle=False``."""

    with np.load(path, allow_pickle=False) as data:
        if int(np.asarray(data["format_version"])[0]) != 1:
            raise ValueError("unsupported parity reference format")
        count = int(np.asarray(data["sample_count"])[0])
        names = tuple(str(value) for value in np.asarray(data["sample_names"]).tolist())
        if len(names) != count:
            raise ValueError("parity reference sample metadata is inconsistent")
        crops = tuple(
            np.asarray(data[f"crop_{index:04d}"]).copy() for index in range(count)
        )
        torch_embeddings = np.asarray(data["torch_embeddings"], dtype=np.float32).copy()
        onnx_embeddings = np.asarray(data["onnx_embeddings"], dtype=np.float32).copy()
    if torch_embeddings.shape != (count, 512) or onnx_embeddings.shape != (count, 512):
        raise ValueError("parity reference embeddings must have shape (N, 512)")
    for name, values in (("Torch", torch_embeddings), ("ONNX", onnx_embeddings)):
        if not np.isfinite(values).all() or not np.allclose(
            np.linalg.norm(values, axis=1), 1.0, atol=1e-3
        ):
            raise ValueError(f"parity reference {name} embeddings are not normalized")
    return names, crops, torch_embeddings, onnx_embeddings


def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a cosine matrix for normalized or otherwise valid embeddings."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 512:
        raise ValueError("embeddings must have shape (N, 512)")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("embeddings must not contain zero vectors")
    normalized = values / norms
    return np.asarray(normalized @ normalized.T, dtype=np.float32)


def _print_matrix(title: str, names: Sequence[str], matrix: np.ndarray) -> None:
    print(title)
    print("samples:", ", ".join(names))
    print(np.array2string(matrix, precision=6, suppress_small=True))


def _print_torch_onnx_report(
    names: Sequence[str],
    torch_embeddings: np.ndarray,
    onnx_embeddings: np.ndarray,
) -> None:
    torch_matrix = pairwise_similarity_matrix(torch_embeddings)
    onnx_matrix = pairwise_similarity_matrix(onnx_embeddings)
    print("Torch vs ONNX per-sample:")
    for name, torch_value, onnx_value in zip(names, torch_embeddings, onnx_embeddings):
        print(
            f"  {name}: cosine={float(np.dot(torch_value, onnx_value)):.8f} "
            f"max_abs_diff={float(np.max(np.abs(torch_value - onnx_value))):.8g}"
        )
    print(
        "Torch/ONNX matrix max_abs_diff=",
        float(np.max(np.abs(torch_matrix - onnx_matrix))),
    )
    _print_matrix("Torch similarity matrix", names, torch_matrix)
    _print_matrix("ONNX similarity matrix", names, onnx_matrix)


def _export(args: argparse.Namespace) -> int:
    import cv2

    from src.ascend_reid import preprocess_reid_crop
    from src.config import load_config
    from src.reid import ReIDExtractor

    config = load_config(args.config)
    crops: list[np.ndarray] = []
    names: list[str] = []
    for crop_path in args.crops:
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            raise FileNotFoundError(f"cannot read crop: {crop_path}")
        crops.append(crop)
        names.append(Path(crop_path).name)

    torch_extractor = ReIDExtractor(
        config.reid,
        args.torch_device,
        backend="torch",
    )
    torch_embeddings = torch_extractor.extract_batch(crops)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "export requires onnxruntime; install it in the PC environment"
        ) from exc
    session = ort.InferenceSession(
        str(Path(args.onnx)),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    if input_name != "images":
        raise ValueError(f"ONNX ReID input must be named 'images', got {input_name!r}")
    output_names = [output.name for output in session.get_outputs()]
    if not output_names or output_names[0] != "embedding":
        raise ValueError(
            "ONNX ReID first output must be named 'embedding', "
            f"got {output_names}"
        )
    onnx_input = np.stack(
        [
            preprocess_reid_crop(
                crop,
                image_height=config.reid.image_height,
                image_width=config.reid.image_width,
            )
            for crop in crops
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    onnx_output = np.asarray(session.run(None, {input_name: onnx_input})[0])
    if onnx_output.shape != (len(crops), 512):
        raise ValueError(f"ONNX output must have shape (N, 512), got {onnx_output.shape}")
    from src.reid import normalize_embedding

    onnx_embeddings = normalize_embedding(onnx_output)
    save_reference(args.output, names, crops, torch_embeddings, onnx_embeddings)
    print(f"saved parity reference: {args.output}")
    _print_torch_onnx_report(names, torch_embeddings, onnx_embeddings)
    return 0


def _check_om(args: argparse.Namespace) -> int:
    from src.ascend_reid import AscendReIDExtractor
    from src.ascend_runtime import AscendRuntime
    from src.config import load_config

    names, crops, torch_embeddings, onnx_embeddings = load_reference(args.reference)
    config = load_config(args.config)
    runtime = AscendRuntime(config.ascend.device_id)
    try:
        extractor = AscendReIDExtractor(
            config.reid,
            config.ascend.reid_model,
            config.ascend.reid_dynamic_batches,
            runtime,
        )
        om_embeddings = extractor.extract_batch(crops)
    finally:
        runtime.close()

    if om_embeddings.shape != torch_embeddings.shape:
        raise ValueError(
            f"OM output shape {om_embeddings.shape} does not match reference "
            f"shape {torch_embeddings.shape}"
        )
    _print_torch_onnx_report(names, torch_embeddings, onnx_embeddings)
    print("Torch vs OM per-sample:")
    for name, torch_value, onnx_value, om_value in zip(
        names,
        torch_embeddings,
        onnx_embeddings,
        om_embeddings,
    ):
        print(
            f"  {name}: torch_om_cosine={float(np.dot(torch_value, om_value)):.8f} "
            f"onnx_om_cosine={float(np.dot(onnx_value, om_value)):.8f} "
            f"torch_om_max_abs_diff={float(np.max(np.abs(torch_value - om_value))):.8g}"
        )
    torch_om_cosines = np.sum(torch_embeddings * om_embeddings, axis=1)
    onnx_om_cosines = np.sum(onnx_embeddings * om_embeddings, axis=1)
    print(
        "Torch vs OM cosine mean/min/max:",
        float(np.mean(torch_om_cosines)),
        float(np.min(torch_om_cosines)),
        float(np.max(torch_om_cosines)),
    )
    print(
        "ONNX vs OM cosine mean/min/max:",
        float(np.mean(onnx_om_cosines)),
        float(np.min(onnx_om_cosines)),
        float(np.max(onnx_om_cosines)),
    )
    torch_matrix = pairwise_similarity_matrix(torch_embeddings)
    onnx_matrix = pairwise_similarity_matrix(onnx_embeddings)
    om_matrix = pairwise_similarity_matrix(om_embeddings)
    print(
        "pairwise matrix max_abs_diff torch/om=",
        float(np.max(np.abs(torch_matrix - om_matrix))),
        "onnx/om=",
        float(np.max(np.abs(onnx_matrix - om_matrix))),
    )
    _print_matrix("OM similarity matrix", names, om_matrix)
    if args.warn_cosine_below is not None:
        for name, value in zip(names, torch_om_cosines):
            if value < args.warn_cosine_below:
                print(
                    f"WARN {name}: Torch-vs-OM cosine {value:.8f} "
                    f"< configured diagnostic bound {args.warn_cosine_below:.8f}"
                )
    if args.warn_max_abs_diff_above is not None:
        for name, torch_value, om_value in zip(names, torch_embeddings, om_embeddings):
            difference = float(np.max(np.abs(torch_value - om_value)))
            if difference > args.warn_max_abs_diff_above:
                print(
                    f"WARN {name}: Torch-vs-OM max_abs_diff {difference:.8g} "
                    f"> configured diagnostic bound {args.warn_max_abs_diff_above:.8g}"
                )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OSNet Torch/ONNX/OM parity diagnostic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="create a PC parity reference")
    export_parser.add_argument("--config", default="config/config.yaml")
    export_parser.add_argument("--onnx", default="deploy/atlas/onnx/osnet_x0_25.onnx")
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--torch-device", default="cpu")
    export_parser.add_argument("crops", nargs="+", type=Path)
    export_parser.set_defaults(handler=_export)

    check_parser = subparsers.add_parser("check-om", help="compare Atlas OM with PC reference")
    check_parser.add_argument("--config", default="config/config_atlas.yaml")
    check_parser.add_argument("--reference", type=Path, required=True)
    check_parser.add_argument(
        "--warn-cosine-below",
        type=float,
        help="optional experiment-specific warning bound; no default is assumed",
    )
    check_parser.add_argument(
        "--warn-max-abs-diff-above",
        type=float,
        help="optional experiment-specific warning bound; no default is assumed",
    )
    check_parser.set_defaults(handler=_check_om)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
