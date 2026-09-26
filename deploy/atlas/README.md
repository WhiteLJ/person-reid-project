# Huawei Atlas 310B deployment

This directory contains the deployment adaptation for Huawei Atlas 310B. It is
an inference-backend variant of the existing MVP-8.3 application, not a new
identity/business MVP. The CPU business logic remains unchanged:

```text
Ascend YOLO OM -> two CPU BoT-SORT states -> Person/Vehicle Track[] -> existing
                 SessionTarget/Recovery/Gallery/SQLite/recognition/enrichment
                 logic
Person crop -> Ascend OSNet OM -> normalized 512-D embedding
Vehicle crop -> Ascend SBS(R50-IBN) OM -> normalized 2048-D embedding
```

The Atlas path does not use `torch_npu`, `YOLO.predict()`, `YOLO.track()`, or a
Torchreid forward pass. Ultralytics 8.4.138 is still used for its CPU BoT-SORT
implementation. BoT-SORT appearance ReID must remain disabled on this path.

## 1. PC: export ONNX

Run this in the already validated PC/PyTorch environment. The export script
uses the current `model.yolo_weight` from the selected YAML configuration. The
repository currently points to `weights/yolo/yolov8n.pt`, so the default output
is `deploy/atlas/onnx/yolov8n.onnx`. If the configured weight is changed to a
YOLO11 checkpoint, the output name follows that checkpoint stem.

The exporter requires the optional PC packages `onnx`; `onnxruntime` is optional
and only used when `--onnx-runtime` is supplied. Do not reinstall or upgrade
the validated PyTorch, CUDA, Ultralytics, or Torchreid packages just to add
these export tools.

```bash
python -m pip install onnx
# Optional CPU sanity check:
python -m pip install onnxruntime

python -m tools.export_atlas_models \
  --config config/config.yaml \
  --onnx-runtime
```

The command exports and validates:

```text
YOLO:  images, 1x3x640x640 -> concrete rank-3 raw detection output, opset 11
OSNet: images, Nx3x256x128 -> embedding, Nx512, opset 11
```

YOLO export is static batch 1, `dynamic=false`, `simplify=false`, and `nms=false`.
The output is raw YOLO data; NMS stays in the CPU postprocessor. OSNet uses a
dynamic batch input and exports features before the classifier head.

The exact default output files are:

```text
deploy/atlas/onnx/yolov8n.onnx
deploy/atlas/onnx/osnet_x0_25.onnx
```

Copy both files to the Atlas board. The ONNX files must be produced by the
validated PC environment; `convert_om.sh` does not export models.

## 2. Atlas: convert ONNX to OM

On Ubuntu 22.04/aarch64 with CANN 6.2.RC2 installed:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod +x deploy/atlas/convert_om.sh

SOC_VERSION=Ascend310B4 ./deploy/atlas/convert_om.sh \
  deploy/atlas/onnx/yolov8n.onnx \
  deploy/atlas/onnx/osnet_x0_25.onnx \
  weights/atlas
```

The script is fail-fast and checks both generated files:

```text
weights/atlas/yolov8n.om
weights/atlas/osnet_x0_25.om
```

The default is `SOC_VERSION=Ascend310B4`; override it without editing source:

```bash
SOC_VERSION=Ascend310B1 ./deploy/atlas/convert_om.sh \
  deploy/atlas/onnx/yolov8n.onnx \
  deploy/atlas/onnx/osnet_x0_25.onnx \
  weights/atlas
```

The script passes YOLO `images:1,3,640,640` and OSNet
`images:-1,3,256,128` with dynamic batches `1,2,4,8`. CANN adds the dynamic
batch control input `ascend_mbatch_shape_data`; the runtime obtains its model
input index and calls the pyACL `set_dynamic_batch_size(model_id, dataset,
index, batch_size)` API before execution. For a candidate list larger than
eight, the runtime executes real chunks such as `8+4+1`; it does not add fake
crops.

The model input name `images` is guaranteed and checked by the PC exporter.
If a manually supplied ONNX file was not produced by that exporter, validate
its input name and shape before conversion rather than changing the ATC command
to guess a different name.

## 3. Atlas Python environment

The board needs:

- Python 3.10 on aarch64;
- CANN 6.2.RC2 runtime and pyACL (`import acl` must work after `set_env.sh`);
- NumPy, OpenCV, and PyYAML;
- Ultralytics 8.4.138 and a CPU-compatible PyTorch installation if required by
  the installed Ultralytics BoT-SORT Python imports.

`requirements.txt` remains the formal PC/Torch project dependency description.
Do not blindly install its PC PyTorch/CUDA packages on Atlas. Use the board's
CANN-compatible Python packages and retain `torch` only when needed for CPU
BoT-SORT imports. Do not install `torch_npu` for this adaptation. The Atlas
neural inference path does not call Torch/Torchreid for YOLO or OSNet.

## 4. Configure and run

Use the checked-in Atlas variant, which keeps the current business thresholds
and changes only the inference backend/model paths:

```yaml
inference:
  backend: ascend

ascend:
  device_id: 0
  yolo_model: weights/atlas/yolov8n.om
  reid_model: weights/atlas/osnet_x0_25.om
  reid_dynamic_batches: [1, 2, 4, 8]
  vehicle_reid_model: weights/atlas/vehicle_sbs_r50_ibn.om
  vehicle_reid_dynamic_batches: [1, 2, 4, 8]

multiclass_tracking:
  vehicle_class_ids: [2, 5, 7]

tracking:
  tracker: config/trackers/botsort_fixed_camera.yaml
```

The corresponding command is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python app.py --config config/config_atlas.yaml --source data/your_video.mp4
```

The application creates one shared ACL runtime and loads YOLO, OSNet, and
Vehicle SBS(R50-IBN) once. Each frame runs Ascend YOLO once, two independent
CPU BoT-SORT updates, and the existing scheduled Person/Vehicle ReID business
logic. The same-frame Person and Vehicle `ReIDFrameCache` instances still
prevent duplicate inference between Recovery and Gallery recognition.

The formal Atlas entry point is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python app.py --config config/config_atlas.yaml --source data/your_video.mp4
```

## 5. Atlas2A Vehicle OM smoke test

Run this on the actual board after the OM files are present:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import acl; print('pyACL import OK')"
python -m tools.atlas_smoke_test \
  --config config/config_atlas.yaml \
  --image data/person.jpg
```

Vehicle multi-class detection/tracking can be checked independently before
running the full application:

```bash
python -m tools.atlas_multiclass_tracking_smoke_test \
  --config config/config_atlas.yaml \
  --source data/your_video.mp4
```

The tool must report `yolo_inference_count == frames` and ends with
`ATLAS_MULTICLASS_TRACKING_SMOKE_OK`.

If YOLO finds no person, supply an image-space crop explicitly:

```bash
python -m tools.atlas_smoke_test \
  --config config/config_atlas.yaml \
  --image data/person.jpg \
  --crop X1 Y1 X2 Y2
```

Expected successful output includes:

```text
pyACL import OK
YOLO detections=<number>
bbox=(...) conf=<number> class=0
OSNet embedding shape=(512,) dtype=float32 norm=1.000000
```

The smoke test must be run on Atlas to validate ACL initialization, OM loading,
dynamic batch selection, and device execution. This PC development environment
does not contain the board's `acl` module, so no hardware result is claimed here.

## 6. Benchmark

Run a fixed regression video and compare the same interval/profile across runs:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m tools.atlas_benchmark \
  --config config/config_atlas.yaml \
  --source data/crowd_regression.mp4 \
  --frames 300
```

The output reports average milliseconds for `YOLO NPU`, `BoT-SORT CPU`,
`OSNet NPU`, `Recovery`, `Gallery`, `Total`, and total FPS. Use `npu-smi info`
while the benchmark is running and expect AICore utilization to become
non-zero during OM inference. A continuously zero reading requires checking
the CANN environment, OM compatibility, and the selected device; it is not a
successful NPU validation.

The benchmark's optional `--reid-samples-per-frame` measures an explicit timing
workload. It does not change the production policy of avoiding OSNet on every
Track on every frame.
