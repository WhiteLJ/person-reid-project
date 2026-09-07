# Person ReID Project - MVP-8.1

This project is an OpenCV prototype for person detection, temporary tracking,
session target recovery, persistent Gallery enrollment, and automatic recognition
of persisted Gallery people.

## Current MVP-8.1 behavior

The runtime pipeline is:

```text
VideoSource -> YOLOv8n person detection -> Ultralytics BoT-SORT
           -> Track[] -> MVP-5 LOST/recovery -> MVP-8 Gallery recognition
           -> OpenCV display
```

The three identity layers remain separate:

- `Track ID`: temporary BoT-SORT trajectory identity;
- `SessionTarget.target_id`: current-process target identity used by LOST/recovery;
- `GalleryPerson.person_id`: persistent logical identity displayed as `P001`, `P002`, ... .

MVP-8.1 supports:

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

The priority in this stage is conservative identity behavior: a valid Track may
remain LOST rather than being rebound to a similar-looking person. The application
does not claim a Track ID change is ground-truth fragmentation, and it does not
automatically classify a recovery as true or false identity without manual video
annotation.

Automatic recognition reads the in-memory Gallery loaded at startup. It does not
write SQLite or update persistent Gallery features. It does not create a new
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

## Configuration

The default configuration is in `config/config.yaml`. Important MVP-8.1 settings are:

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
  recovery_min_track_age_frames: 3
  recovery_confirmation_hits: 2
  recovery_pending_max_age_frames: 60

reid_quality:
  min_track_confidence: 0.35
  max_edge_truncation_ratio: 0.30
  max_person_overlap_ratio: 0.50

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
- `C`: clear all current SessionTargets and their runtime associations;
- `Enter`/`Space`: finish an edit session;
- `Q`: exit the entire application, including from an edit session.

`C` removes all TARGET special boxes. With the default
`show_unselected_tracks: false`, ordinary green Track boxes still remain hidden;
set that option to `true` to show them.

## Offline Gallery administration

The lightweight MVP-7/MVP-8.1 development tool supports listing and deleting
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

## Tests

The normal test suite uses fake ReID extractors and temporary SQLite files; it does
not require a camera, GUI, GPU inference, or network downloads:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Not implemented yet

MVP-8.1 does not include PySide/Qt UI, RTSP, face recognition, training/fine-tuning,
BoxMOT, automatic persistent Gallery feature updates, or complex Gallery editing.
False recovery counts require manual review against the fixed regression video; the
program reports attempts, pending proposals, accepted recoveries, and Track
created/ended events only. The next stage may address live-stream robustness and
later presentation/UI work.
