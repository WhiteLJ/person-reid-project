# Huawei Atlas 310B deployment

This directory contains the deployment adaptation for Huawei Atlas 310B. It is
an inference-backend variant of the existing MVP-8.2 application, not a new
identity/business MVP. The CPU business logic remains unchanged:

```text
Ascend YOLO OM -> CPU BoT-SORT -> Track[] -> existing SessionTarget/Recovery/
                 Gallery/SQLite/recognition/enrichment logic
Track crop -> Ascend OSNet OM -> normalized 512-D embedding
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
- Pillow (used to reproduce Torchreid's PIL preprocessing);
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

tracking:
  tracker: config/trackers/botsort_baseline.yaml
```

The corresponding command is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python app.py --config config/config_atlas.yaml --source data/your_video.mp4
```

The application creates one shared ACL runtime, loads YOLO and OSNet once, and
reuses them. Each frame runs Ascend YOLO once, CPU BoT-SORT once, and the
existing scheduled ReID business logic. The same-frame `ReIDFrameCache` still
prevents duplicate OSNet inference between Recovery and Gallery recognition.

## 5. Smoke test

Run this on the actual board after the OM files are present:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import acl; print('pyACL import OK')"
python -m tools.atlas_smoke_test \
  --config config/config_atlas.yaml \
  --image data/person.jpg
```

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

## 6. Latency/jitter benchmark

Run the same fixed regression video and interval for every comparison:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m tools.atlas_benchmark \
  --config config/config_atlas.yaml \
  --source data/crowd_regression.mp4 \
  --frames 300 \
  --frame-log reports/atlas_benchmark.jsonl
```

The output reports `average`, `p50`, `p95`, `p99`, and `max` milliseconds for:

```text
video_read, yolo_preprocess, yolo_h2d, yolo_npu, yolo_d2h,
yolo_decode_nms, botsort, osnet_preprocess, osnet_npu,
recovery, gallery_recognition, render, imshow_waitKey, total
```

The JSONL file records each frame's Gallery candidate count, OSNet invocation
count, actual dynamic batch sizes, and stage timings. A periodic spike such as
40 candidates -> `8+8+8+8+8` is therefore visible instead of being hidden in an
average. `osnet_preprocess` and `osnet_npu` are recorded per actual extractor
call; `total` remains per video frame.

Use `--display` only when measuring GUI overhead. The optional
`--reid-samples-per-frame N` adds an explicit diagnostic OSNet workload and does
not change production scheduling.

The runtime reuses device buffers, datasets, data buffers, and host output
storage. Repeated execution should not call `acl.rt.malloc`,
`acl.mdl.create_dataset`, or their matching free/destroy functions. OSNet
workspaces are keyed by real batch size. To test batch 16, first regenerate the
OSNet OM with dynamic batches including 16, then set the YAML value to
`[1, 2, 4, 8, 16]`; do not merely change YAML for an OM that does not support it.

Run `npu-smi info` while the benchmark is running and expect AICore utilization
to become non-zero during OM inference. A continuously zero reading requires
checking CANN, OM compatibility, and device selection; it is not a successful
NPU validation.

## 7. Controlled BoT-SORT A/B

Keep the video, target, frame interval, and Gallery database fixed. Run a fresh
process for each row and change one variable at a time:

```text
baseline profile:     config/trackers/botsort_baseline.yaml
fixed-camera profile: config/trackers/botsort_crowd_fixed.yaml
```

The fixed-camera profile uses `gmc_method: none`, a supported value in the
installed Ultralytics 8.4.138 GMC implementation. Use it only for a fixed
camera. Compare tracker latency/FPS and manually annotated Track
fragmentation/identity errors. Separately test `conf_threshold` 0.35, 0.20,
0.15; `track_buffer` 30, 60, 90; and `image_size` 640, 960. Do not combine
these into an uncontrolled grid or call any value universally optimal.

The `with_reid: true, model: auto` profile is retained as an experiment, but
the Atlas adapter deliberately rejects it so it cannot start an extra PyTorch
appearance model on the board. If evaluated on the Torch/PC path, record it
separately; it is not part of the formal Atlas neural-inference path.

## 8. Torch -> ONNX -> OM ReID parity

On the PC, use exactly the same crops for Torchreid and ONNX Runtime:

```bash
python -m tools.reid_backend_parity export \
  --config config/config.yaml \
  --onnx deploy/atlas/onnx/osnet_x0_25.onnx \
  --torch-device cpu \
  --output deploy/atlas/parity/reid_parity_reference.npz \
  data/reid_crops/A1.jpg data/reid_crops/A2.jpg data/reid_crops/B1.jpg
```

The tool stores crops and both normalized embeddings in a non-pickle NPZ. It
prints per-sample Torch/ONNX cosine and max absolute difference plus both
pairwise similarity matrices. The ONNX input uses the production Atlas
preprocessing helper; the Torch side uses the official Torchreid transform.

Copy the NPZ to Atlas and run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m tools.reid_backend_parity check-om \
  --config config/config_atlas.yaml \
  --reference deploy/atlas/parity/reid_parity_reference.npz
```

This prints Torch-vs-OM, ONNX-vs-OM, and pairwise-matrix differences. Optional
`--warn-cosine-below` and `--warn-max-abs-diff-above` are diagnostic bounds
only; no universal correctness threshold is assumed. A same-crop cosine
materially below 1 or changed same-person/different-person ordering is evidence
to inspect preprocessing, ONNX export, ATC/SOC compatibility, or output
decoding before changing recognition thresholds.

## 9. Gallery feature diagnosis

Inspect the persisted Gallery without changing it:

```bash
python -m tools.gallery_reid_diagnose --config config/config_atlas.yaml
```

It reports each person's reference count, centroid norm, pairwise reference
cosines, near-duplicate pair count, and reference-to-centroid statistics. The
enrichment-only diversity starting value is
`gallery_enrichment.reference_duplicate_threshold: 0.98`; it does not alter
SessionTarget Recovery or recognition scoring. Production recognition remains
candidate-vs-centroid, not max historical-reference score.

For explicit candidate-crop diagnosis:

```bash
python -m tools.gallery_reid_diagnose \
  --config config/config_atlas.yaml \
  --candidate-crop debug/candidate_001.jpg debug/candidate_002.jpg \
  --track-ids 17 18
```

With debug logging enabled, the application reports top-1/top-2 person scores
and margins. Combine this with bbox/crop inspection; a Track ID change alone is
not ground truth for a true or false recovery.

## 10. Interpreting Gallery regression cases

Compare both using the same fixed video interval:

1. Existing Torch-created SQLite Gallery -> Atlas OM recognition.
2. Empty test Gallery -> Atlas S/G enrollment -> restart -> Atlas OM recognition.

If case 1 fails but case 2 works, first investigate Torch/ONNX/OM feature-space
parity and the old feature bank. If both fail, investigate candidate crops,
feature diversity, and conservative matching. If parity is high but both cases
fail, do not blame OM conversion without evidence; inspect ranking, quality
gates, and Gallery contents.

No command in this document claims that the Atlas board has been tested by the
PC test suite. Hardware smoke, benchmark, and `npu-smi` results must be captured
on the actual board.
