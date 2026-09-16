from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.ascend_runtime import AscendRuntime
from src.ascend_runtime import AscendRuntimeError


class _FakeUtil:
    @staticmethod
    def bytes_to_ptr(value):
        return value

    @staticmethod
    def numpy_to_ptr(value):
        return value


class _FakeRT:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_device(self, _device_id):
        self.calls.append("set_device")
        return 0

    def create_context(self, _device_id):
        self.calls.append("create_context")
        return "context", 0

    def malloc(self, size, _policy):
        self.calls.append("malloc")
        return bytearray(size), 0

    def memcpy(self, destination, destination_size, source, source_size, _kind):
        self.calls.append("memcpy")
        del source_size
        if isinstance(destination, bytearray):
            source_bytes = (
                source.tobytes()
                if isinstance(source, np.ndarray)
                else bytes(source)
            )
            destination[:destination_size] = source_bytes[:destination_size]
        elif isinstance(destination, np.ndarray):
            destination[:] = np.frombuffer(
                bytes(source)[:destination_size], dtype=destination.dtype
            )
        return 0

    def free(self, _pointer):
        self.calls.append("free")
        return 0

    def destroy_context(self, _context):
        self.calls.append("destroy_context")
        return 0

    def reset_device(self, _device_id):
        self.calls.append("reset_device")
        return 0


class _FakeMdl:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.execute_status = 0
        self.load_status = 0

    def load_from_file(self, _path):
        self.calls.append("load_from_file")
        return "model", self.load_status

    def create_desc(self):
        self.calls.append("create_desc")
        return "desc"

    def get_desc(self, _desc, _model):
        self.calls.append("get_desc")
        return 0

    def get_num_inputs(self, _desc):
        return 2

    def get_num_outputs(self, _desc):
        return 1

    def get_input_size_by_index(self, _desc, _index):
        return 4

    def get_output_size_by_index(self, _desc, _index):
        return 4

    def get_input_dims(self, _desc, _index):
        return {"dims": [1, 1]}

    def get_output_dims(self, _desc, _index):
        return {"dims": [1]}

    def create_dataset(self):
        self.calls.append("create_dataset")
        return []

    def get_input_index_by_name(self, _desc, name):
        self.calls.append(f"get_input_index_by_name:{name}")
        return 1, 0

    def add_dataset_buffer(self, dataset, data_buffer):
        dataset.append(data_buffer)
        return 0

    def execute(self, _model, _inputs, outputs):
        if self.execute_status:
            return self.execute_status
        output_pointer = outputs[0]["ptr"]
        output_pointer[:4] = np.asarray((1.0,), dtype=np.float32).tobytes()
        return 0

    def destroy_dataset(self, _dataset):
        self.calls.append("destroy_dataset")
        return 0

    def destroy_desc(self, _desc):
        self.calls.append("destroy_desc")
        return 0

    def unload(self, _model):
        self.calls.append("unload")
        return 0

    def set_dynamic_batch_size(self, _model, _dataset, _index, _batch_size):
        self.calls.append("set_dynamic_batch_size")
        return 0


class _FakeAcl:
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2
    ACL_MEM_MALLOC_HUGE_FIRST = 3

    def __init__(self) -> None:
        self.rt = _FakeRT()
        self.mdl = _FakeMdl()
        self.util = _FakeUtil()
        self.init_calls = 0
        self.finalize_calls = 0
        self.destroy_buffer_calls = 0

    def init(self):
        self.init_calls += 1
        return 0

    def finalize(self):
        self.finalize_calls += 1
        return 0

    @staticmethod
    def create_data_buffer(pointer, size):
        return {"ptr": pointer, "size": size}

    def get_data_buffer_addr(self, data_buffer):
        return data_buffer["ptr"]

    def destroy_data_buffer(self, _data_buffer):
        self.destroy_buffer_calls += 1
        return 0


class _FakeAclWithoutMemoryConstants(_FakeAcl):
    """Simulate a pyACL build that omits top-level memory aliases."""

    _MISSING = {
        "ACL_MEM_MALLOC_HUGE_FIRST",
        "ACL_MEMCPY_HOST_TO_DEVICE",
        "ACL_MEMCPY_DEVICE_TO_HOST",
    }

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, "_MISSING"):
            raise AttributeError(name)
        return super().__getattribute__(name)


class AscendRuntimeTests(unittest.TestCase):
    def _model_path(self, name: str) -> Path:
        model_path = Path(".test_tmp") / name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"OM")
        return model_path

    def test_fixed_model_reuses_workspace_across_many_executes(self) -> None:
        acl = _FakeAcl()
        runtime = AscendRuntime(0, acl_module=acl)
        model = runtime.load_model(self._model_path("fake_runtime_reuse.om"))

        for _ in range(100):
            outputs = runtime.execute(
                model,
                [
                    np.ones((1,), dtype=np.float32),
                    np.ones((1,), dtype=np.float32),
                ],
            )

        self.assertAlmostEqual(float(outputs[0][0]), 1.0)
        self.assertEqual(acl.rt.calls.count("malloc"), 3)
        self.assertEqual(acl.mdl.calls.count("create_dataset"), 2)
        self.assertEqual(acl.destroy_buffer_calls, 0)

        runtime.close()
        self.assertEqual(acl.destroy_buffer_calls, 3)
        self.assertEqual(acl.rt.calls.count("free"), 3)
        self.assertEqual(acl.mdl.calls.count("destroy_dataset"), 2)
        runtime.close()
        self.assertEqual(acl.destroy_buffer_calls, 3)

    def test_dynamic_batch_sequence_reuses_one_workspace_per_batch(self) -> None:
        acl = _FakeAcl()
        runtime = AscendRuntime(0, acl_module=acl)
        model = runtime.load_model(
            self._model_path("fake_runtime_dynamic_reuse.om"),
            dynamic_batch_sizes=(1, 2, 4, 8),
        )

        for batch_size in (1, 1, 4, 8, 4, 1):
            runtime.execute(
                model,
                [np.ones((batch_size, 1), dtype=np.float32)],
                dynamic_batch=batch_size,
            )

        # Each batch workspace owns one data input, one dynamic selector input,
        # and one output buffer.  The repeated 1/4 batches must not allocate.
        self.assertEqual(acl.rt.calls.count("malloc"), 3 * 3)
        self.assertEqual(acl.mdl.calls.count("create_dataset"), 3 * 2)
        self.assertEqual(acl.mdl.calls.count("set_dynamic_batch_size"), 6)

        runtime.close()
        self.assertEqual(acl.destroy_buffer_calls, 3 * 3)
        self.assertEqual(acl.rt.calls.count("free"), 3 * 3)
        self.assertEqual(acl.mdl.calls.count("destroy_dataset"), 3 * 2)
        runtime.close()
        self.assertEqual(acl.destroy_buffer_calls, 3 * 3)

    def test_runtime_load_execute_and_close_lifecycle_is_reusable(self) -> None:
        model_path = Path(".test_tmp") / "fake_runtime.om"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"OM")
        acl = _FakeAcl()

        runtime = AscendRuntime(0, acl_module=acl)
        model = runtime.load_model(model_path, dynamic_batch_sizes=(1, 2))
        self.assertEqual(model.dynamic_batch_input_index, 1)
        self.assertEqual(model.data_input_indices, (0,))
        outputs = runtime.execute(
            model,
            [np.ones((1,), dtype=np.float32)],
            dynamic_batch=1,
        )
        runtime.close()
        runtime.close()

        self.assertEqual(outputs[0].shape, (1,))
        self.assertAlmostEqual(float(outputs[0][0]), 1.0)
        self.assertEqual(acl.init_calls, 1)
        self.assertEqual(acl.finalize_calls, 1)
        self.assertEqual(acl.mdl.calls.count("load_from_file"), 1)
        self.assertEqual(acl.mdl.calls.count("unload"), 1)
        self.assertEqual(acl.mdl.calls.count("destroy_desc"), 1)
        self.assertEqual(acl.rt.calls.count("destroy_context"), 1)
        self.assertEqual(acl.rt.calls.count("reset_device"), 1)
        self.assertEqual(acl.mdl.calls.count("set_dynamic_batch_size"), 1)

    def test_acl_execute_error_is_raised_and_runtime_can_be_closed(self) -> None:
        model_path = Path(".test_tmp") / "fake_runtime_error.om"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"OM")
        acl = _FakeAcl()
        runtime = AscendRuntime(0, acl_module=acl)
        model = runtime.load_model(model_path)
        acl.mdl.execute_status = 107

        with self.assertRaises(AscendRuntimeError) as context:
            runtime.execute(
                model,
                [
                    np.ones((1,), dtype=np.float32),
                    np.ones((1,), dtype=np.float32),
                ],
            )

        self.assertIn("107", str(context.exception))
        runtime.close()

    def test_execute_failure_keeps_cached_workspace_cleanable_once(self) -> None:
        acl = _FakeAcl()
        runtime = AscendRuntime(0, acl_module=acl)
        model = runtime.load_model(self._model_path("fake_runtime_cached_error.om"))

        acl.mdl.execute_status = 107
        with self.assertRaises(AscendRuntimeError):
            runtime.execute(
                model,
                [
                    np.ones((1,), dtype=np.float32),
                    np.ones((1,), dtype=np.float32),
                ],
            )

        runtime.close()
        self.assertEqual(acl.destroy_buffer_calls, 3)
        runtime.close()
        self.assertEqual(acl.destroy_buffer_calls, 3)

    def test_failed_model_load_preserves_primary_error_without_invalid_unload(self) -> None:
        model_path = Path(".test_tmp") / "fake_runtime_load_error.om"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"OM")
        acl = _FakeAcl()
        acl.mdl.load_status = 145002

        runtime = AscendRuntime(0, acl_module=acl)
        with self.assertRaises(AscendRuntimeError) as context:
            runtime.load_model(model_path)

        message = str(context.exception)
        self.assertIn("145002", message)
        self.assertIn(str(model_path.resolve()), message)
        self.assertEqual(acl.mdl.calls.count("unload"), 0)
        runtime.close()

    def test_missing_pyacl_memory_aliases_use_documented_integer_values(self) -> None:
        model_path = Path(".test_tmp") / "fake_runtime_without_aliases.om"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"OM")
        acl = _FakeAclWithoutMemoryConstants()

        runtime = AscendRuntime(0, acl_module=acl)
        model = runtime.load_model(model_path)
        outputs = runtime.execute(
            model,
            [
                np.ones((1,), dtype=np.float32),
                np.ones((1,), dtype=np.float32),
            ],
        )
        runtime.close()

        self.assertAlmostEqual(float(outputs[0][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
