from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import onnx
except ImportError:  # pragma: no cover - optional PC export dependency
    onnx = None

from tools.export_atlas_models import build_parser, validate_onnx


@unittest.skipIf(onnx is None, "ONNX export dependency is not installed")
class VehicleAtlasExportContractTests(unittest.TestCase):
    def test_parser_requires_explicit_vehicle_opt_in(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.include_vehicle_reid)
        self.assertEqual(
            defaults.vehicle_reid_output,
            Path("deploy/atlas/onnx/vehicle_sbs_r50_ibn.onnx"),
        )
        enabled = parser.parse_args(["--include-vehicle-reid"])
        self.assertTrue(enabled.include_vehicle_reid)

    def test_vehicle_onnx_contract_validator_accepts_dynamic_batch(self) -> None:
        input_info = onnx.helper.make_tensor_value_info(
            "images", onnx.TensorProto.FLOAT, ["batch", 3, 256, 256]
        )
        output_info = onnx.helper.make_tensor_value_info(
            "embedding", onnx.TensorProto.FLOAT, ["batch", 2048]
        )
        node = onnx.helper.make_node("Identity", ["images"], ["embedding"])
        graph = onnx.helper.make_graph(
            [node],
            "vehicle_contract",
            [input_info],
            [output_info],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 11)],
            ir_version=7,
        )

        path = Path("vehicle_contract.onnx")
        with patch("onnx.load", return_value=model):
            input_shape, output_shapes = validate_onnx(
                path,
                expected_input_shape=(None, 3, 256, 256),
                expected_output_width=2048,
                dynamic_batch=True,
                expected_output_name="embedding",
            )

        self.assertEqual(input_shape, (None, 3, 256, 256))
        self.assertEqual(output_shapes, ((None, 2048),))


if __name__ == "__main__":
    unittest.main()
