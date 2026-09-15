"""Export the validated Torch models to ONNX for Atlas ATC conversion.

This script runs on the PC/PyTorch environment.  It deliberately performs no
ATC conversion; that step belongs on the Atlas board with its installed CANN.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import load_config


def _shape_from_value(value: Any) -> tuple[int | None, ...]:
    dimensions = []
    for dimension in value:
        if hasattr(dimension, "dim_value") and dimension.dim_value:
            dimensions.append(int(dimension.dim_value))
        else:
            dimensions.append(None)
    return tuple(dimensions)


def validate_onnx(
    path: Path,
    *,
    expected_input_shape: tuple[int | None, ...],
    expected_output_width: int | None,
    dynamic_batch: bool,
) -> tuple[tuple[int | None, ...], tuple[tuple[int | None, ...], ...]]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX validation requires the PC export dependency 'onnx'"
        ) from exc

    model = onnx.load(str(path))
    opsets = [opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")]
    if opsets != [11]:
        raise RuntimeError(f"{path} must use exactly ONNX opset 11, found {opsets}")
    if len(model.graph.input) != 1 or model.graph.input[0].name != "images":
        names = [value.name for value in model.graph.input]
        raise RuntimeError(f"{path} must have one input named 'images', found {names}")

    input_shape = _shape_from_value(model.graph.input[0].type.tensor_type.shape.dim)
    if len(input_shape) != len(expected_input_shape):
        raise RuntimeError(f"{path} input rank is invalid: {input_shape}")
    for index, (actual, expected) in enumerate(zip(input_shape, expected_input_shape)):
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"{path} input dimension {index} must be {expected}, found {actual}"
            )
        if expected is None and not dynamic_batch and actual is None:
            raise RuntimeError(f"{path} has an unexpected dynamic input dimension")
    if dynamic_batch and input_shape[0] is None:
        pass
    elif dynamic_batch:
        raise RuntimeError(f"{path} ReID batch dimension is not dynamic: {input_shape}")

    if not model.graph.output:
        raise RuntimeError(f"{path} has no outputs")
    output_shapes = tuple(
        _shape_from_value(output.type.tensor_type.shape.dim)
        for output in model.graph.output
    )
    first_output = output_shapes[0]
    if expected_output_width is None:
        if len(first_output) != 3 or any(value is None for value in first_output):
            raise RuntimeError(
                f"{path} YOLO output must be a concrete rank-3 tensor, found {first_output}"
            )
    elif len(first_output) != 2 or first_output[1] != expected_output_width:
        raise RuntimeError(
            f"{path} first output must be (?, {expected_output_width}), found {first_output}"
        )
    return input_shape, output_shapes


def _copy_exported_model(exported: Any, destination: Path) -> Path:
    source = Path(str(exported))
    if not source.is_file():
        raise FileNotFoundError(f"Ultralytics/Torch export did not create {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def export_yolo(config: Any, destination: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("YOLO ONNX export requires the installed ultralytics package") from exc

    if not config.model.yolo_weight.is_file():
        raise FileNotFoundError(f"YOLO weight file not found: {config.model.yolo_weight}")
    model = YOLO(str(config.model.yolo_weight))
    exported = model.export(
        format="onnx",
        opset=11,
        imgsz=config.model.image_size,
        batch=1,
        dynamic=False,
        simplify=False,
        nms=False,
        device="cpu",
    )
    output = _copy_exported_model(exported, destination)
    input_shape, output_shapes = validate_onnx(
        output,
        expected_input_shape=(1, 3, config.model.image_size, config.model.image_size),
        expected_output_width=None,
        dynamic_batch=False,
    )
    print(f"YOLO ONNX: {output} input={input_shape} outputs={output_shapes}")
    return output


def export_reid(config: Any, destination: Path) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("OSNet ONNX export requires PyTorch") from exc

    from src.reid import ReIDExtractor

    extractor = ReIDExtractor(config.reid, "cpu", backend="torch")
    feature_model = extractor._feature_extractor.model

    class FeatureOnly(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            output = self.model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 2 or output.shape[1] != 512:
                raise RuntimeError(f"OSNet model output is not (N,512): {tuple(output.shape)}")
            return output

    wrapped = FeatureOnly(feature_model).eval()
    dummy = torch.zeros(
        (1, 3, config.reid.image_height, config.reid.image_width),
        dtype=torch.float32,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(destination),
        opset_version=11,
        input_names=["images"],
        output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        do_constant_folding=True,
    )
    input_shape, output_shapes = validate_onnx(
        destination,
        expected_input_shape=(
            None,
            3,
            config.reid.image_height,
            config.reid.image_width,
        ),
        expected_output_width=512,
        dynamic_batch=True,
    )
    print(f"OSNet ONNX: {destination} input={input_shape} outputs={output_shapes}")
    return destination


def onnx_runtime_sanity(path: Path, shape: tuple[int, ...]) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not installed; skipped optional runtime sanity check")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sample = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, {input_name: sample})
    print(f"ONNX Runtime sanity: {path.name} output_shapes={[np.asarray(o).shape for o in outputs]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export project YOLO and OSNet models to ONNX opset 11")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--yolo-output", type=Path)
    parser.add_argument("--reid-output", type=Path, default=Path("deploy/atlas/onnx/osnet_x0_25.onnx"))
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-reid", action="store_true")
    parser.add_argument("--onnx-runtime", action="store_true", help="run optional ONNX Runtime CPU sanity checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    yolo_output = args.yolo_output or Path("deploy/atlas/onnx") / (
        config.model.yolo_weight.stem + ".onnx"
    )
    if not args.skip_yolo:
        exported = export_yolo(config, yolo_output)
        if args.onnx_runtime:
            onnx_runtime_sanity(
                exported,
                (1, 3, config.model.image_size, config.model.image_size),
            )
    if not args.skip_reid:
        exported = export_reid(config, args.reid_output)
        if args.onnx_runtime:
            onnx_runtime_sanity(
                exported,
                (1, 3, config.reid.image_height, config.reid.image_width),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
