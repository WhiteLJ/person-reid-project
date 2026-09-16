"""Shared pyACL/AscendCL runtime for Atlas OM inference.

The module intentionally imports ``acl`` only when an Ascend runtime is
constructed.  This keeps the PC/Torch backend importable on machines that do
not have CANN installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class AscendRuntimeError(RuntimeError):
    """Raised when an AscendCL operation fails."""


def _status(value: Any) -> int:
    """Extract an ACL status from a normal integer or ``(value, status)``."""

    if isinstance(value, tuple) and value and isinstance(value[-1], int):
        return int(value[-1])
    if isinstance(value, int):
        return int(value)
    return 0


def _handle_and_status(value: Any, operation: str) -> tuple[Any, int]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
        return value[0], int(value[1])
    if value is None:
        raise AscendRuntimeError(f"{operation} returned no handle")
    # create_desc/create_dataset commonly return a handle directly.  Model
    # loading and malloc return a (handle, ret) tuple on pyACL.
    return value, 0


def _check(result: Any, operation: str) -> None:
    code = _status(result)
    if code != 0:
        hints = {
            145002: (
                "the OM model path is invalid or the file is not accessible; "
                "verify the resolved path, permissions, file integrity, and SOC_VERSION"
            ),
            145003: "the model ID is invalid, often as a secondary error after a failed load",
        }
        hint = hints.get(code)
        suffix = f" ({hint})" if hint else ""
        raise AscendRuntimeError(
            f"{operation} failed with ACL error code {code}{suffix}"
        )


def _acl_constant(acl_module: Any, name: str, default: int) -> int:
    """Read a pyACL constant, supporting CANN builds that omit Python aliases.

    Some CANN/pyACL releases expose these values as module attributes while
    others document them as integer constants that applications define.  The
    values are part of the AscendCL API contract, so use the documented
    fallback when the Python alias is absent.
    """

    value = getattr(acl_module, name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AscendRuntimeError(
            f"pyACL constant {name} must be an integer, got {value!r}"
        ) from exc


@dataclass
class AscendModel:
    """Loaded OM model metadata owned by one :class:`AscendRuntime`."""

    path: Path
    model_id: Any
    description: Any
    input_count: int
    output_count: int
    input_sizes: tuple[int, ...]
    output_sizes: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...] | None, ...] = ()
    output_shapes: tuple[tuple[int, ...] | None, ...] = ()
    dynamic_batch_sizes: tuple[int, ...] = ()
    data_input_indices: tuple[int, ...] = ()
    dynamic_batch_input_index: int | None = None
    workspaces: dict[int | None, "_ExecutionWorkspace"] = field(default_factory=dict)


@dataclass
class _ExecutionWorkspace:
    """Reusable device/dataset resources for one model and batch shape."""

    input_dataset: Any | None = None
    output_dataset: Any | None = None
    input_pointers: dict[int, Any] = field(default_factory=dict)
    input_buffers: dict[int, Any] = field(default_factory=dict)
    output_pointers: list[Any] = field(default_factory=list)
    output_buffers: list[Any] = field(default_factory=list)
    input_sizes: dict[int, int] = field(default_factory=dict)
    host_outputs: list[np.ndarray] = field(default_factory=list)
    released: bool = False


class AscendRuntime:
    """Own one ACL process/device/context lifecycle and reusable OM models."""

    def __init__(self, device_id: int = 0, acl_module: Any | None = None) -> None:
        if isinstance(device_id, bool) or device_id < 0:
            raise ValueError("Ascend device_id must be non-negative")
        self.device_id = device_id
        self.acl = acl_module if acl_module is not None else import_module("acl")
        self._models: list[AscendModel] = []
        self._closed = False
        self._device_set = False
        self._context: Any | None = None
        self._acl_initialized = False

        try:
            _check(self.acl.init(), "acl.init")
            self._acl_initialized = True
            _check(self.acl.rt.set_device(device_id), "acl.rt.set_device")
            self._device_set = True
            context_value = self.acl.rt.create_context(device_id)
            self._context, ret = _handle_and_status(
                context_value, "acl.rt.create_context"
            )
            _check(ret, "acl.rt.create_context")
        except Exception:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def load_model(
        self,
        model_path: str | Path,
        *,
        dynamic_batch_sizes: Sequence[int] = (),
    ) -> AscendModel:
        """Load an OM once and query its descriptor sizes."""

        if self._closed:
            raise AscendRuntimeError("cannot load an OM after runtime close")
        # ACL model loading is more reliable with a canonical absolute path,
        # especially when the application is launched through a shell or
        # service with a different working directory.
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Ascend OM model not found: {path}")
        if path.stat().st_size == 0:
            raise AscendRuntimeError(f"Ascend OM model is empty: {path}")

        model_id: Any | None = None
        model_loaded = False
        description: Any | None = None
        try:
            load_from_file = getattr(self.acl.mdl, "load_from_file", None)
            if load_from_file is None:
                raise AscendRuntimeError(
                    "installed pyACL does not expose acl.mdl.load_from_file"
                )
            model_id_value = load_from_file(str(path))
            model_id, ret = _handle_and_status(
                model_id_value,
                "acl.mdl.load_from_file",
            )
            _check(ret, "acl.mdl.load_from_file")
            model_loaded = True

            description_value = self.acl.mdl.create_desc()
            description, ret = _handle_and_status(
                description_value, "acl.mdl.create_desc"
            )
            _check(ret, "acl.mdl.create_desc")
            _check(
                self.acl.mdl.get_desc(description, model_id),
                "acl.mdl.get_desc",
            )

            input_count = int(self.acl.mdl.get_num_inputs(description))
            output_count = int(self.acl.mdl.get_num_outputs(description))
            input_sizes = tuple(
                int(self.acl.mdl.get_input_size_by_index(description, index))
                for index in range(input_count)
            )
            output_sizes = tuple(
                int(self.acl.mdl.get_output_size_by_index(description, index))
                for index in range(output_count)
            )
            input_shapes = tuple(
                self._query_dims(description, index, kind="input")
                for index in range(input_count)
            )
            output_shapes = tuple(
                self._query_dims(description, index, kind="output")
                for index in range(output_count)
            )
            batches = tuple(int(value) for value in dynamic_batch_sizes)
            if any(value < 1 for value in batches):
                raise ValueError("dynamic batch sizes must be positive")
            dynamic_batch_input_index = None
            if batches:
                dynamic_batch_input_index = self._get_input_index_by_name(
                    description,
                    "ascend_mbatch_shape_data",
                )
                if not 0 <= dynamic_batch_input_index < input_count:
                    raise AscendRuntimeError(
                        "dynamic batch input index is outside the model input range: "
                        f"index={dynamic_batch_input_index} inputs={input_count}"
                    )
            data_input_indices = tuple(
                index
                for index in range(input_count)
                if index != dynamic_batch_input_index
            )
            model = AscendModel(
                path=path,
                model_id=model_id,
                description=description,
                input_count=input_count,
                output_count=output_count,
                input_sizes=input_sizes,
                output_sizes=output_sizes,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                dynamic_batch_sizes=batches,
                data_input_indices=data_input_indices,
                dynamic_batch_input_index=dynamic_batch_input_index,
            )
            self._models.append(model)
            return model
        except Exception as operation_error:
            cleanup_errors: list[Exception] = []
            if description is not None:
                try:
                    _check(
                        self.acl.mdl.destroy_desc(description),
                        "acl.mdl.destroy_desc",
                    )
                except Exception as exc:
                    cleanup_errors.append(exc)
            # A failed load may return an implementation-defined/invalid handle.
            # Do not call unload in that case: it masks the primary load error
            # with ACL_ERROR_GE_EXEC_MODEL_ID_INVALID (145003).
            if model_loaded and model_id is not None:
                try:
                    _check(self.acl.mdl.unload(model_id), "acl.mdl.unload")
                except Exception as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise AscendRuntimeError(
                    "Ascend model load failed and rollback cleanup also failed: "
                    f"path={path}; operation={operation_error}; cleanup={cleanup_errors[0]}"
                ) from operation_error
            if isinstance(operation_error, AscendRuntimeError):
                raise AscendRuntimeError(
                    f"{operation_error} (path={path})"
                ) from operation_error
            raise

    def execute(
        self,
        model: AscendModel,
        inputs: Sequence[np.ndarray],
        *,
        dynamic_batch: int | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Execute an OM using a reusable single-threaded ACL workspace.

        A workspace is keyed by the dynamic batch size.  The first execution
        of a fixed model or batch shape pays the allocation/dataset setup cost;
        subsequent executions only copy data, execute, and copy outputs.
        """

        if self._closed:
            raise AscendRuntimeError("cannot execute after runtime close")
        if model not in self._models:
            raise AscendRuntimeError("model does not belong to this runtime")
        data_input_indices = (
            model.data_input_indices
            if model.data_input_indices
            else tuple(range(model.input_count))
        )
        if len(inputs) != len(data_input_indices):
            raise ValueError(
                f"model expects {len(data_input_indices)} data inputs, got {len(inputs)}"
            )
        if dynamic_batch is not None:
            if dynamic_batch < 1:
                raise ValueError("dynamic_batch must be positive")
            if model.dynamic_batch_input_index is None:
                raise AscendRuntimeError(
                    "dynamic_batch was requested for a model without "
                    "ascend_mbatch_shape_data"
                )
            if model.dynamic_batch_sizes and dynamic_batch not in model.dynamic_batch_sizes:
                raise ValueError(
                    f"dynamic batch {dynamic_batch} is not supported; "
                    f"choose one of {model.dynamic_batch_sizes}"
                )

        arrays = [
            np.ascontiguousarray(np.asarray(value, dtype=np.float32))
            for value in inputs
        ]
        workspace_key = (
            dynamic_batch if model.dynamic_batch_input_index is not None else None
        )
        workspace = model.workspaces.get(workspace_key)
        if workspace is None:
            workspace = self._create_workspace(
                model,
                arrays,
                data_input_indices,
                dynamic_batch,
            )
            model.workspaces[workspace_key] = workspace
        else:
            self._validate_workspace_inputs(workspace, data_input_indices, arrays)

        for index, array in zip(data_input_indices, arrays):
            pointer = workspace.input_pointers[index]
            _check(
                self.acl.rt.memcpy(
                    pointer,
                    array.nbytes,
                    self.acl.util.numpy_to_ptr(array),
                    array.nbytes,
                    _acl_constant(self.acl, "ACL_MEMCPY_HOST_TO_DEVICE", 1),
                ),
                "acl.rt.memcpy host-to-device",
            )

        if dynamic_batch is not None:
            self._set_dynamic_batch_size(
                model,
                workspace.input_dataset,
                dynamic_batch,
            )

        _check(
            self.acl.mdl.execute(
                model.model_id,
                workspace.input_dataset,
                workspace.output_dataset,
            ),
            "acl.mdl.execute",
        )

        output_values: list[np.ndarray] = []
        for index, (data_buffer, output_size, host_buffer) in enumerate(
            zip(workspace.output_buffers, model.output_sizes, workspace.host_outputs)
        ):
            device_ptr = self.acl.get_data_buffer_addr(data_buffer)
            _check(
                self.acl.rt.memcpy(
                    self.acl.util.numpy_to_ptr(host_buffer),
                    output_size,
                    device_ptr,
                    output_size,
                    _acl_constant(self.acl, "ACL_MEMCPY_DEVICE_TO_HOST", 2),
                ),
                "acl.rt.memcpy device-to-host",
            )
            # Preserve the existing execute() ownership contract: callers may
            # retain returned outputs while the workspace is reused by the next
            # inference.  The host allocation itself is reused; only this
            # caller-owned result copy remains necessary.
            output = host_buffer.copy()
            shape = model.output_shapes[index] if model.output_shapes else None
            if shape is not None and all(value > 0 for value in shape):
                expected_values = int(np.prod(shape))
                if expected_values == output.size:
                    output = output.reshape(shape)
            output_values.append(output)
        return tuple(output_values)

    def close(self) -> None:
        """Release models, context, device, and ACL exactly once."""

        if self._closed:
            return
        first_error: Exception | None = None
        for model in reversed(self._models):
            for workspace in list(model.workspaces.values()):
                try:
                    self._release_workspace(workspace)
                except Exception as exc:
                    first_error = first_error or exc
            model.workspaces.clear()
            try:
                _check(
                    self.acl.mdl.destroy_desc(model.description),
                    "acl.mdl.destroy_desc",
                )
            except Exception as exc:
                first_error = first_error or exc
            try:
                _check(self.acl.mdl.unload(model.model_id), "acl.mdl.unload")
            except Exception as exc:
                first_error = first_error or exc
        self._models.clear()

        if self._context is not None:
            try:
                _check(
                    self.acl.rt.destroy_context(self._context),
                    "acl.rt.destroy_context",
                )
            except Exception as exc:
                first_error = first_error or exc
            self._context = None
        if self._device_set:
            try:
                _check(self.acl.rt.reset_device(self.device_id), "acl.rt.reset_device")
            except Exception as exc:
                first_error = first_error or exc
            self._device_set = False
        if self._acl_initialized:
            try:
                _check(self.acl.finalize(), "acl.finalize")
            except Exception as exc:
                first_error = first_error or exc
            self._acl_initialized = False
        self._closed = True
        if first_error is not None:
            raise AscendRuntimeError(f"Ascend runtime cleanup failed: {first_error}")

    def _create_workspace(
        self,
        model: AscendModel,
        arrays: Sequence[np.ndarray],
        data_input_indices: Sequence[int],
        dynamic_batch: int | None,
    ) -> _ExecutionWorkspace:
        """Create one persistent workspace, rolling back partial setup."""

        workspace = _ExecutionWorkspace()
        try:
            workspace.input_dataset = self._create_dataset("input")
            for index, array in zip(data_input_indices, arrays):
                pointer = self._malloc(array.nbytes)
                workspace.input_pointers[index] = pointer
                workspace.input_sizes[index] = array.nbytes
                data_buffer = self.acl.create_data_buffer(pointer, array.nbytes)
                if data_buffer is None:
                    raise AscendRuntimeError(
                        "acl.create_data_buffer returned no input buffer"
                    )
                workspace.input_buffers[index] = data_buffer

            if model.dynamic_batch_input_index is not None:
                index = model.dynamic_batch_input_index
                pointer = self._malloc(model.input_sizes[index])
                workspace.input_pointers[index] = pointer
                workspace.input_sizes[index] = model.input_sizes[index]
                data_buffer = self.acl.create_data_buffer(
                    pointer,
                    model.input_sizes[index],
                )
                if data_buffer is None:
                    raise AscendRuntimeError(
                        "acl.create_data_buffer returned no dynamic-batch buffer"
                    )
                workspace.input_buffers[index] = data_buffer

            for index in sorted(workspace.input_buffers):
                _check(
                    self.acl.mdl.add_dataset_buffer(
                        workspace.input_dataset,
                        workspace.input_buffers[index],
                    ),
                    f"acl.mdl.add_dataset_buffer input[{index}]",
                )

            workspace.output_dataset = self._create_dataset("output")
            for output_size in model.output_sizes:
                if output_size % np.dtype(np.float32).itemsize != 0:
                    raise AscendRuntimeError(
                        f"output byte size {output_size} is not float32-aligned"
                    )
                pointer = self._malloc(output_size)
                workspace.output_pointers.append(pointer)
                data_buffer = self.acl.create_data_buffer(pointer, output_size)
                if data_buffer is None:
                    raise AscendRuntimeError(
                        "acl.create_data_buffer returned no output buffer"
                    )
                workspace.output_buffers.append(data_buffer)
                workspace.host_outputs.append(
                    np.empty(
                        (output_size // np.dtype(np.float32).itemsize,),
                        dtype=np.float32,
                    )
                )
                _check(
                    self.acl.mdl.add_dataset_buffer(
                        workspace.output_dataset,
                        data_buffer,
                    ),
                    "acl.mdl.add_dataset_buffer output",
                )
            return workspace
        except BaseException as operation_error:
            try:
                self._release_workspace(workspace)
            except Exception as cleanup_error:
                raise AscendRuntimeError(
                    "Ascend workspace creation failed and rollback cleanup also "
                    f"failed: operation={operation_error}; cleanup={cleanup_error}"
                ) from operation_error
            raise

    def _validate_workspace_inputs(
        self,
        workspace: _ExecutionWorkspace,
        data_input_indices: Sequence[int],
        arrays: Sequence[np.ndarray],
    ) -> None:
        expected = {
            index: array.nbytes
            for index, array in zip(data_input_indices, arrays)
        }
        actual = {
            index: workspace.input_sizes[index]
            for index in data_input_indices
        }
        if expected != actual:
            raise ValueError(
                "input byte sizes changed for a cached Ascend workspace; "
                f"expected={actual} got={expected}"
            )

    def _release_workspace(self, workspace: _ExecutionWorkspace) -> None:
        """Release every resource in one workspace at most once."""

        if workspace.released:
            return
        first_error: Exception | None = None
        for data_buffer in list(workspace.input_buffers.values()) + list(
            workspace.output_buffers
        ):
            try:
                _check(
                    self.acl.destroy_data_buffer(data_buffer),
                    "acl.destroy_data_buffer",
                )
            except Exception as exc:
                first_error = first_error or exc
        for dataset, name in (
            (workspace.input_dataset, "input"),
            (workspace.output_dataset, "output"),
        ):
            if dataset is not None:
                try:
                    _check(
                        self.acl.mdl.destroy_dataset(dataset),
                        f"acl.mdl.destroy_dataset {name}",
                    )
                except Exception as exc:
                    first_error = first_error or exc
        for pointer in list(workspace.input_pointers.values()) + list(
            workspace.output_pointers
        ):
            try:
                _check(self.acl.rt.free(pointer), "acl.rt.free")
            except Exception as exc:
                first_error = first_error or exc
        workspace.input_pointers.clear()
        workspace.input_buffers.clear()
        workspace.output_pointers.clear()
        workspace.output_buffers.clear()
        workspace.input_sizes.clear()
        workspace.host_outputs.clear()
        workspace.input_dataset = None
        workspace.output_dataset = None
        workspace.released = True
        if first_error is not None:
            raise AscendRuntimeError(
                f"Ascend workspace cleanup failed: {first_error}"
            ) from first_error

    def _create_dataset(self, name: str) -> Any:
        value = self.acl.mdl.create_dataset()
        dataset, ret = _handle_and_status(value, f"acl.mdl.create_dataset {name}")
        _check(ret, f"acl.mdl.create_dataset {name}")
        if dataset is None:
            raise AscendRuntimeError(f"acl.mdl.create_dataset {name} returned no dataset")
        return dataset

    def _query_dims(
        self,
        description: Any,
        index: int,
        *,
        kind: str,
    ) -> tuple[int, ...] | None:
        getter = getattr(self.acl.mdl, f"get_{kind}_dims", None)
        if getter is None:
            return None
        try:
            value = getter(description, index)
        except Exception as exc:
            raise AscendRuntimeError(
                f"acl.mdl.get_{kind}_dims failed for index {index}"
            ) from exc
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
            value, ret = value
            _check(ret, f"acl.mdl.get_{kind}_dims")
        if isinstance(value, dict):
            value = value.get("dims")
        elif hasattr(value, "dims"):
            value = value.dims
        if value is None:
            return None
        try:
            dimensions = tuple(int(dimension) for dimension in value)
        except (TypeError, ValueError) as exc:
            raise AscendRuntimeError(
                f"acl.mdl.get_{kind}_dims returned invalid dimensions: {value!r}"
            ) from exc
        return dimensions or None

    def _malloc(self, size: int) -> Any:
        value = self.acl.rt.malloc(
            size,
            _acl_constant(self.acl, "ACL_MEM_MALLOC_HUGE_FIRST", 0),
        )
        pointer, ret = _handle_and_status(value, "acl.rt.malloc")
        _check(ret, "acl.rt.malloc")
        if pointer is None:
            raise AscendRuntimeError("acl.rt.malloc returned no device pointer")
        return pointer

    def _set_dynamic_batch_size(
        self,
        model: AscendModel,
        input_dataset: Any,
        batch_size: int,
    ) -> None:
        """Set CANN's dynamic-batch selector for ``ascend_mbatch_shape_data``."""

        setter = getattr(self.acl.mdl, "set_dynamic_batch_size", None)
        if setter is None:
            raise AscendRuntimeError(
                "installed pyACL does not expose acl.mdl.set_dynamic_batch_size"
            )
        _check(
            setter(
                model.model_id,
                input_dataset,
                model.dynamic_batch_input_index,
                batch_size,
            ),
            "acl.mdl.set_dynamic_batch_size",
        )

    def _get_input_index_by_name(self, description: Any, name: str) -> int:
        getter = getattr(self.acl.mdl, "get_input_index_by_name", None)
        if getter is None:
            raise AscendRuntimeError(
                "installed pyACL does not expose acl.mdl.get_input_index_by_name; "
                f"cannot configure dynamic input {name!r}"
            )
        value = getter(description, name)
        index, ret = _handle_and_status(value, "acl.mdl.get_input_index_by_name")
        _check(ret, "acl.mdl.get_input_index_by_name")
        try:
            return int(index)
        except (TypeError, ValueError) as exc:
            raise AscendRuntimeError(
                f"acl.mdl.get_input_index_by_name returned invalid index: {index!r}"
            ) from exc

    def __enter__(self) -> "AscendRuntime":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()
