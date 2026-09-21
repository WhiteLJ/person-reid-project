# Person ReID Project - MVP-8.2

This project is an OpenCV prototype for person detection, temporary tracking,
session target recovery, persistent Gallery enrollment, and automatic recognition
of persisted Gallery people.

## Current MVP-8.2 behavior

The runtime pipeline is:

```text
VideoSource -> YOLOv8n person detection -> Ultralytics BoT-SORT
           -> Track[] -> MVP-5 LOST/recovery -> accepted reference events
           -> MVP-8.2 safe Gallery enrichment -> MVP-8 Gallery recognition
           -> OpenCV display
```

The three identity layers remain separate:

- `Track ID`: temporary BoT-SORT trajectory identity;
- `SessionTarget.target_id`: current-process target identity used by LOST/recovery;
- `GalleryPerson.person_id`: persistent logical identity displayed as `P001`, `P002`, ... .

The MVP-8.1 foundation supports:

- camera and local video input;
- YOLOv8 person detection and BoT-SORT temporary Track IDs;
- multi-target manual selection and LOST -> ReID recovery;
- explicit Gallery enrollment with SQLite persistence;
- automatic, conservative recognition of loaded Gallery people using normalized
  512-D Torchreid/OSNet embeddings;
- centroid similarity, threshold/margin checks, one-to-one matching, and repeated
  confirmation before automatic binding.

Crowded-scene safeguards add:

- Recovery candidate Track age and two-attempt confirmation before a LOST target is
  rebound;
- a configurable ReID quality gate for confidence, crop size, frame-edge truncation,
  and person-bbox overlap;
- same-frame ReID embedding reuse across Recovery and Gallery recognition;
- baseline and experimental crowd BoT-SORT profiles under `config/trackers/`;
- lightweight Track lifecycle, recovery, recognition, quality-rejection, and FPS
  diagnostics.

MVP-8.2 adds safe persistent Gallery feature enrichment:

- S then G can create a Gallery person immediately, even with one reference;
- later accepted ACTIVE runtime references can update that explicitly enrolled
  person's bounded representative Gallery bank without another OSNet inference;
- the SessionTarget runtime bank remains an 8-entry FIFO for short-term recovery,
  while the persistent Gallery bank is maintained independently and does not use
  FIFO eviction;
- the enrollment-time reference is a protected anchor, near-duplicate references
  are rejected, and a full bank replaces only a more redundant non-anchor when the
  new reference adds more diversity;
- each accepted reference update is consumed once, and SQLite feature replacement
  is atomic before the in-memory snapshot is replaced;
- after recovery, enrichment waits for the configured stable ACTIVE cooldown;
- a person recognized automatically from the persisted Gallery is not enriched
  until the user explicitly presses G for that current SessionTarget.

Enrichment is single-threaded and does not change the SQLite schema. LOST targets,
recovery candidates, rejected/low-quality references, and pending recovery attempts
never update persistent Gallery features.

The priority in this stage is conservative identity behavior: a valid Track may
remain LOST rather than being rebound to a similar-looking person. The application
does not claim a Track ID change is ground-truth fragmentation, and it does not
automatically classify a recovery as true or false identity without manual video
annotation.

Automatic recognition reads the in-memory Gallery loaded at startup. Recognition
itself does not write SQLite, update persistent Gallery features, or create a new
`GalleryPerson` during recognition.

Ordinary unselected Tracks remain tracked in the background. By default they are
not drawn. Set this option to `true` for green debugging boxes:

```yaml
ui:
  show_unselected_tracks: false
```

Selected or automatically recognized active targets are drawn in red, for example
`TARGET P001 | ID 17`. A target without a Gallery identity is shown as
`TARGET | ID 17`.

## Environment and weights

The formal installation entry point is `requirements.txt`; `pip-freeze.txt` is only
a local environment snapshot. Torchreid is sourced from the official
[`KaiyangZhou/deep-person-reid`](https://github.com/KaiyangZhou/deep-person-reid)
Git repository, not from the unrelated/old PyPI package with the same name.

Do not upgrade the already validated PyTorch, CUDA, or Ultralytics environment for
this MVP. Before running the application, prepare:

```text
weights/yolo/yolov8n.pt
weights/reid/osnet_x0_25_msmt17.pth
```

The ReID checkpoint is the official OSNet x0.25 MSMT17 combineall checkpoint. The
application fails clearly when it is missing and never silently falls back to
ImageNet-only or random weights.

## Atlas 310B inference backend

The project also contains a deployment-only Atlas adaptation. It does not change
the MVP-8.2 identity or persistence behavior. Set `inference.backend: ascend` in
`config/config_atlas.yaml` to run YOLO and OSNet from locally converted OM models
through pyACL/AscendCL. BoT-SORT remains the installed Ultralytics CPU
implementation, and the project does not use `torch_npu`.

The PC default remains `inference.backend: torch`. The Atlas workflow is fully
documented in [`deploy/atlas/README.md`](deploy/atlas/README.md): export both
models to ONNX opset 11 on the PC, convert them with ATC on the board, then run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python app.py --config config/config_atlas.yaml --source data/your_video.mp4
```

The current repository model is `weights/yolo/yolov8n.pt`, so the default export
and OM names are `yolov8n.onnx` and `yolov8n.om`. The exporter uses the configured
weight path, so a later configured YOLO11 checkpoint produces the corresponding
model stem without changing the backend code. `SOC_VERSION` defaults to
`Ascend310B4` in the conversion script and can be overridden for another 310B
variant. Atlas-specific ONNX/OM tools are deployment tooling, not additional
runtime business dependencies; `requirements.txt` remains the formal PC
dependency entry point.

## Configuration

The default configuration is in `config/config.yaml`. Important MVP-8.2 settings are:

```yaml
gallery_recognition:
  enabled: true
  recognition_interval_frames: 10
  min_track_age_frames: 5
  recognition_threshold: 0.80
  recognition_margin: 0.05
  confirmation_hits: 2

database:
  path: "database/person_reid.db"

reid_recovery:
  # Conservative engineering starting values based on the current test video;
  # these are not universal optima.
  recovery_threshold: 0.85
  recovery_reference_support_threshold: 0.80
  recovery_reference_support_top_k: 3
  recovery_min_track_age_frames: 3
  recovery_confirmation_hits: 2
  recovery_pending_max_age_frames: 60

gallery_enrichment:
  # Engineering starting value, not a universal optimum.
  post_recovery_stable_frames: 30
  max_reference_embeddings: 8
  duplicate_similarity_threshold: 0.97

reid_quality:
  min_track_confidence: 0.35
  max_edge_truncation_ratio: 0.30
  max_person_overlap_ratio: 0.60
  min_frame_edge_margin_ratio: 0.01

diagnostics:
  enabled: true
  log_interval_frames: 300
```

These are conservative, configurable engineering starting values, not universal
thresholds. Recognition uses the in-memory Gallery and does not query SQLite per
frame.

### Crowd A/B experiments

Keep the regression video, target, and time range fixed and change one variable at
a time. The project profiles are:

- `config/trackers/botsort_baseline.yaml`: installed BoT-SORT defaults,
  `with_reid: false`, `track_buffer: 30`;
- `config/trackers/botsort_crowd.yaml`: experiment profile,
  `with_reid: false`, `track_buffer: 60`;
- `config/trackers/botsort_crowd_reid.yaml`: separate optional appearance-ReID
  experiment, `with_reid: true`, `model: auto`.

For each profile, vary detection `model.conf_threshold` across `0.35`, `0.20`,
`0.15`, tracker `track_buffer` across `30`, `60`, `90`, and `image_size` across
`640`, `960` as separate experiments. Test the appearance-ReID profile last. These
settings are not declared universally optimal and should not be combined into an
uncontrolled grid for MVP-8.1.

`ActiveIdentityGuard` is not part of the MVP-8.1 main chain. The observed failure
mode is Track fragmentation followed by false LOST recovery, so no extra per-frame
ACTIVE verification is enabled in this stage.

## MVP-8.3-PC1 Vehicle ReID

MVP-8.3-PC1 is an independent PC-only Vehicle ReID validation tool. It does not
connect Vehicle ReID to detection, tracking, SessionTarget, Recovery, Gallery,
SQLite, or Atlas. The project-owned configuration is
`config/vehicle_reid/sbs_R50-ibn.yml`, based on the official FastReID VeRi
`SBS(R50-ibn)` configuration.

The official checkpoint is:

```text
https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth
```

Download it to `weights/vehicle_reid/` before running the smoke test. PowerShell:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth" `
  -OutFile "weights/vehicle_reid/veri_sbs_R50-ibn.pth"
```

Linux/macOS:

```bash
curl -L \
  -o weights/vehicle_reid/veri_sbs_R50-ibn.pth \
  https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth
```

The standalone extractor uses the official VeRi SBS R50-IBN inference structure,
including IBN, Non-local blocks, Generalized Mean Pooling, and BN neck. It accepts
BGR crops, converts them to RGB, resizes with cubic interpolation to `256x256`,
and applies the FastReID ImageNet normalization. The true output dimension is
reported after the first real model inference; it is not treated as the Person
OSNet 512-D contract.

Run the PC smoke test with three vehicle crops/frames:

```bash
python -m tools.vehicle_reid_smoke_test \
  --config config/config.yaml \
  --image-a path/to/vehicle_a_frame1.jpg \
  --image-b path/to/vehicle_a_frame2.jpg \
  --image-c path/to/vehicle_b.jpg
```

The current implementation adds no new runtime dependency and does not alter the
Person ReID or Atlas paths. FastReID remains a read-only reference under
`references/fast-reid`; formal project code does not import from that directory.

## MVP-8.3-PC2: Person + Vehicle tracking smoke test

PC2 is an independent PC-only validation pipeline. It performs one YOLO
prediction per frame for COCO `person` (class `0`) and `car` (class `2`), then
sends the two detection groups to separate CPU BoT-SORT instances. It does not
connect Vehicle ReID to SessionTarget, Recovery, Gallery, SQLite, or Atlas.

The class split is configured under `multiclass_tracking.vehicle_class_ids` in
`config/config.yaml`. The temporary tracker namespaces are intentionally shown
as `P-T{id}` and `V-T{id}`; a numeric Person Track ID and Vehicle Track ID may
be equal because they belong to different tracker instances.

Run it with a video containing people and cars:

```bash
python -m tools.multiclass_tracking_smoke_test \
  --config config/config.yaml \
  --source data/vehicle_pc2/test.mp4
```

Press `Q` to stop. On exit the tool reports processed frames, average FPS,
unique Person/Vehicle Track counts, and the YOLO inference count. For a run of
`N` processed frames, the inference count must also be `N`.

## MVP-8.3-PC4A: Vehicle LOST and incremental Recovery

PC4A is a PC-only calibration/infrastructure stage. It reuses the shared
SessionTarget Recovery core with an independent Vehicle TargetManager,
Vehicle ReID frame cache, 2048-D Vehicle embeddings, and a Vehicle-only quality
gate. The default PC tracker profile is
`config/trackers/botsort_fixed_camera.yaml`, because the supported cameras are
fixed and the profile disables BoT-SORT GMC while keeping `with_reid: false`.

Vehicle Recovery uses an incremental sweep. The initial
`vehicle_recovery.recovery_candidates_per_frame: 1` is intentionally
conservative and the Vehicle thresholds are engineering placeholders only;
calibrate them before judging real recovery accuracy.

Calibrate same-vehicle/different-vehicle similarity distributions with one
directory per vehicle:

```bash
python -m tools.vehicle_reid_calibration \
  --config config/config.yaml \
  --root data/vehicle_calibration
```

Run the interactive PC4A smoke test with `S`, `R`, `C`, and `Q`:

```bash
python -m tools.vehicle_recovery_smoke_test \
  --config config/config.yaml \
  --source data/vehicle_pc4/test.mp4
```

PC4A does not implement Vehicle Gallery, SQLite, restart recognition, Atlas,
or Vehicle persistence. Do not treat the initial Vehicle Recovery thresholds as
calibrated until the calibration output has been reviewed.

## Run

Use the default camera:

```bash
python app.py
```

Use a local video:

```bash
python app.py --source data/demo.mp4
```

Controls:

- `S`: pause and add one or more current Tracks as SessionTargets;
- `R`: pause and remove one or more current SessionTargets;
- `G`: pause and explicitly enroll existing SessionTargets into the Gallery;
  for an automatically recognized target this grants enrichment for the current
  session and keeps the existing person ID;
- `C`: clear all current SessionTargets and their runtime associations;
- `Enter`/`Space`: finish an edit session;
- `Q`: exit the entire application, including from an edit session.

`C` removes all TARGET special boxes. With the default
`show_unselected_tracks: false`, ordinary green Track boxes still remain hidden;
set that option to `true` to show them.

## Offline Gallery administration

The lightweight MVP-7/MVP-8.2 development tool supports listing and deleting
persistent Gallery people:

```bash
python -m tools.gallery_admin list
python -m tools.gallery_admin remove P001
python -m tools.gallery_admin clear
python -m tools.gallery_admin --db path/to/test.db list
```

This is an offline management tool. Close the running main application before
`remove` or `clear`, otherwise its in-memory `TargetGallery` can differ from the
SQLite database until restart. Removing a Gallery person does not delete an
existing in-memory SessionTarget in a running application.

To inspect persistent reference diversity without running inference or changing
the database:

```bash
python -m tools.gallery_reference_diagnose --config config/config.yaml --person P001
```

The tool prints the reference count, centroid-to-reference similarities, and the
full reference-to-reference cosine matrix.

## Tests

The normal test suite uses fake ReID extractors and temporary SQLite files; it does
not require a camera, GUI, GPU inference, or network downloads:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Not implemented yet

MVP-8.2 does not include PySide/Qt UI, RTSP, face recognition, training/fine-tuning,
BoxMOT, unrestricted automatic/adaptive Gallery feature updates, or complex Gallery
editing. Automatically recognized targets remain excluded from persistent enrichment
until explicit G enrollment in the current run.
False recovery counts require manual review against the fixed regression video; the
program reports attempts, pending proposals, accepted recoveries, and Track
created/ended events only. The next stage may address live-stream robustness and
later presentation/UI work.
