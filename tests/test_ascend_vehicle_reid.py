from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.ascend_runtime import AscendRuntimeError
from src.ascend_vehicle_reid import AscendVehicleReIDExtractor
from src.config import VehicleReIDConfig


class _FakeRuntime:
    def __init__(self, output_factory=None, error=None) -> None:
        self.calls: list[tuple[int | None, tuple[int, ...], str]] = []
        self.output_factory = output_factory
        self.error = error

    def load_model(self, _path, *, dynamic_batch_sizes):
        self.loaded_batches = tuple(dynamic_batch_sizes)
        return object()

    def execute(self, _model, inputs, *, dynamic_batch):
        chunk = inputs[0]
        self.calls.append((dynamic_batch, chunk.shape, str(chunk.dtype)))
        if self.error is not None:
            raise self.error
        if self.output_factory is not None:
            return (self.output_factory(chunk, dynamic_batch),)
        output = np.zeros((dynamic_batch, 2048), dtype=np.float32)
        output[:, 0] = 1.0
        return (output,)


def _config() -> VehicleReIDConfig:
    return VehicleReIDConfig(
        enabled=True,
        model_name="vehicle",
        weight=Path("unused.pth"),
        config=Path("unused.yml"),
        device="cpu",
        image_height=256,
        image_width=256,
    )


def _crops(count: int) -> list[np.ndarray]:
    return [np.full((30, 40, 3), index + 1, dtype=np.uint8) for index in range(count)]


class AscendVehicleReIDTests(unittest.TestCase):
    def _extractor(self, runtime: _FakeRuntime) -> AscendVehicleReIDExtractor:
        return AscendVehicleReIDExtractor(
            _config(),
            "weights/atlas/vehicle_sbs_r50_ibn.om",
            (1, 2, 4, 8),
            runtime,
            model=object(),
        )

    def test_empty_batch_has_2048_dimension(self) -> None:
        result = self._extractor(_FakeRuntime()).extract_batch([])
        self.assertEqual(result.shape, (0, 2048))
        self.assertEqual(result.dtype, np.float32)

    def test_dynamic_chunks_and_inference_count(self) -> None:
        runtime = _FakeRuntime()
        extractor = self._extractor(runtime)
        for count, expected in ((1, [1]), (2, [2]), (4, [4]), (8, [8]), (3, [2, 1]), (5, [4, 1]), (7, [4, 2, 1])):
            runtime.calls.clear()
            result = extractor.extract_batch(_crops(count))
            self.assertEqual(result.shape, (count, 2048))
            self.assertEqual([call[0] for call in runtime.calls], expected)
            self.assertTrue(np.allclose(np.linalg.norm(result, axis=1), 1.0))
            self.assertTrue(all(call[1] == (size, 3, 256, 256) for call, size in zip(runtime.calls, expected)))
        self.assertEqual(extractor.inference_count, 1 + 1 + 1 + 1 + 2 + 2 + 3)

    def test_invalid_output_dimension_is_rejected(self) -> None:
        runtime = _FakeRuntime(output_factory=lambda _chunk, _batch: np.zeros((1, 512), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "not divisible by 2048"):
            self._extractor(runtime).extract_batch(_crops(1))

    def test_nan_output_is_rejected(self) -> None:
        def output(_chunk, batch):
            result = np.ones((batch, 2048), dtype=np.float32)
            result[0, 0] = np.nan
            return result

        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            self._extractor(_FakeRuntime(output_factory=output)).extract_batch(_crops(1))

    def test_runtime_error_contains_execution_context(self) -> None:
        runtime = _FakeRuntime(error=AscendRuntimeError("ACL error code 123"))
        with self.assertRaisesRegex(
            AscendRuntimeError,
            r"model_path=.*vehicle_sbs_r50_ibn\.om.*batch_size=1.*input_shape=.*input_dtype=float32.*ACL error code 123",
        ):
            self._extractor(runtime).extract_batch(_crops(1))


if __name__ == "__main__":
    unittest.main()
