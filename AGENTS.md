# AGENTS.md

## 1. Project Goal

This repository implements an engineering/competition prototype for interactive person tracking and person re-identification.

Required end-to-end behavior:

1. Detect people from video/camera/RTSP.
2. Track each visible person with a temporary Track ID.
3. Allow the user to manually select any visible person.
4. Keep that person highlighted while visible.
5. If the person leaves the frame and later returns, recover the target using Person ReID even if the Track ID changes.
6. Allow the user to explicitly enroll the current target into a persistent Target Gallery.
7. Automatically recognize enrolled people when they appear again.
8. Persist enrolled Person IDs and embeddings across application restarts.

This is an engineering project, not a research project. Prefer mature libraries and simple reliable code over algorithmic novelty.

---

## 2. Source of Truth

The main design document is:

```text
person_reid_project_spec.md
```

Follow that document unless the user explicitly changes the requirements.

Do not silently redesign the architecture.

---

## 3. Required Architecture

Preferred first-version stack:

```text
Ultralytics YOLOv8 (`yolov8n.pt`)
    +
Ultralytics BoT-SORT
    +
Torchreid 1.4.0 / OSNet (`osnet_x0_25`)
    +
TargetManager
    +
TargetGallery
    +
SQLite
    +
OpenCV UI
```

BoxMOT is a fallback/advanced tracker reference, not a mandatory first-version dependency.

The staged implementation boundary is important: MVP-1 uses YOLOv8 with `yolov8n.pt`
for person-only detection and OpenCV display. BoT-SORT starts in MVP-2; Torchreid/
OSNet, TargetGallery, and SQLite start in later MVPs and must not be pulled into MVP-1
runtime behavior.

In MVP-2, the main loop must reuse one `TrackingPipeline` instance. That instance loads
one YOLO model and calls `model.track(..., persist=True, tracker=<configured BoT-SORT
profile>)` once per frame. Do not run `predict()` and `track()` on the same frame, and
do not pass `workers` for NumPy-frame tracking input.

In MVP-3, manual selection is multi-target: `TargetManager` stores a set of temporary
Track IDs. ROI selection and removal must use the current frame's existing `tracks`
list and IoU matching; they must not trigger another YOLO/BoT-SORT inference. MVP-3
must not load or call Torchreid/OSNet, create Person IDs, use TargetGallery, or use
SQLite.

In MVP-3.1, the OpenCV UI uses a mouse-callback edit session on the existing main
window. ADD_TARGETS and REMOVE_TARGETS must freeze the current frame and Track list,
allow multiple ROI edits before Enter/Space, and clear the callback in a `finally`
block. Unselected Tracks remain tracked but are hidden by default; Q from an edit
session must propagate to application shutdown.

In MVP-5, the authoritative selection state is a set of in-memory `SessionTarget`
records, not a mutable selected-Track-ID set. Each record keeps a temporary session
`target_id`, `current_track_id`, `last_track_id`, ACTIVE/LOST state, a bounded
normalized reference bank, and centroid. User selection must create the initial ReID
reference from the selected crop; invalid crops must not create a recoverable target.
The main loop may create one OSNet/ReID extractor, update ACTIVE references only at a
configured interval after a centroid-consistency check, and attempt recovery only for
due LOST targets and unbound candidate Tracks. Recovery must use batch ReID, a
configurable threshold and explicitly defined two-sided margin, with one-to-one
assignments. MVP-5 still does not implement Person ID, TargetGallery, or SQLite.

In MVP-6, `TargetGallery` is an in-memory, explicit-enrollment domain module. It owns
`GalleryPerson` records and the `session_target_id -> person_id` association lifecycle.
G may enroll only an existing SessionTarget and must copy its reference bank and
centroid without invoking ReID. R/C must detach target mappings without deleting
GalleryPerson records; removing a GalleryPerson must remove all reverse mappings while
leaving SessionTargets intact. MVP-6 must not add persistence or automatic Gallery
recognition.

In MVP-7, SQLite persistence is isolated in `src/database.py` as a
`GalleryRepository`; SQL must not be added to `TargetGallery`. Only GalleryPerson
fields (person_id, label, reference embeddings, and centroid) are persisted. Each
connection must execute `PRAGMA foreign_keys = ON`, embeddings use validated float32
512-D BLOBs without pickle, and the database parent directory is created from the
configured relative path. The person records, all embeddings, and the monotonic
`gallery_meta.next_person_id` update must commit in one transaction. SessionTarget,
Track, runtime state, and session-target mappings are never persisted or restored.
Startup loads GalleryPerson records before video processing; it does not automatically
recognize ordinary Tracks. G enrollment must persist only a newly created person, and
must roll back that new in-memory person if persistence fails without damaging an
existing duplicate enrollment. Gallery remove/clear operations must coordinate memory
and disk without silently leaving them inconsistent. `tools/gallery_admin.py` is an
offline management tool, so the main application must be closed before remove/clear.

In MVP-8, `src/gallery_recognition.py` may automatically match ordinary current
person Tracks against the in-memory Gallery loaded at startup. It must not query or
write SQLite during recognition, enroll new GalleryPerson records, or update
persistent Gallery features. Recovery runs before Gallery recognition; only Tracks
successfully claimed by Recovery, currently owned by an ACTIVE SessionTarget, or
otherwise occupied are excluded. A Track merely inspected and rejected by Recovery
remains eligible, and a shared `ReIDFrameCache` may reuse its embedding only within
the same frame index. Recognition uses centroid similarity, configurable threshold
and two-sided margin, deterministic one-to-one matching, and configurable repeated
confirmation before binding. A loaded GalleryPerson is occupied while attached to
either an ACTIVE or LOST SessionTarget. An already selected but Gallery-unbound
SessionTarget must be attached in place rather than replaced by a second target.
Track-age and recognition-interval controls limit ReID work; no Gallery recognition
is performed when the Gallery is empty or no eligible candidates exist.

In MVP-8.1, prioritize crowded-scene Track fragmentation and false LOST recovery.
Do not treat a Track ID change as confirmed fragmentation without ground truth, and
do not add an ACTIVE identity guard to the main chain; the current observed failure
mode is Track loss followed by an incorrect recovery. Recovery candidates must be
continuously tracked for a configurable minimum age, pass a configurable ReID
quality gate, and require repeated valid attempts for the same target/Track pair
before recovery. A quality-rejected frame is not an identity mismatch and must not
clear recovery pending state; candidate disappearance, a later valid failed attempt,
or a changed candidate may clear it. The gate must consider confidence, crop size,
frame-edge truncation, and the maximum intersection-over-target-area overlap with
other person boxes. Recovery-inspected but unmatched Tracks remain eligible for
Gallery recognition and may reuse only the current frame's `ReIDFrameCache` entry;
successfully recovered or otherwise occupied Tracks remain excluded.

MVP-8.1 keeps the default BoT-SORT profile as a baseline and provides project-owned
crowd A/B profiles under `config/trackers/`. Detection confidence, tracker buffer,
image size, and optional BoT-SORT appearance ReID are tested as controlled
single-variable experiments; no profile is declared universally optimal. Runtime
diagnostics report Track created/ended events, target LOST/RECOVERED events, ReID
attempt/pending/accepted counts, quality rejections, Gallery recognitions, and
average FPS. The application must not automatically label a recovery true/false.

In MVP-8.2, explicitly G-enrolled SessionTargets may safely enrich their mapped
GalleryPerson using only accepted runtime reference-update events from the existing
MVP-5/8.1 pipeline; enrichment never runs an extra ReID inference. Reference-update
events are drained once, and rejected references, LOST targets, recovery candidates,
and recovery-pending states never update persistent Gallery features. Initial S->G
enrollment is immediate, while a target must remain stably ACTIVE for the configured
post-recovery cooldown before enrichment resumes after recovery. A target recognized
automatically from the persisted Gallery is not eligible until the user explicitly G
enrolls it in the current process. Repository feature replacement is an atomic
SQLite transaction, followed by applying the same validated deep-copied snapshot to
memory; any unexpected apply failure must be logged and repaired from Repository.
MVP-8.2 remains single-threaded, does not change the SQLite schema, and does not
automatically update Gallery features for automatically recognized targets.

---

## 4. Critical Identity Rule

Never confuse Track ID with Person ID.

```text
Track ID  = temporary tracker identity
Person ID = persistent application identity
```

A person may return with a different Track ID and still keep the same Person ID.

Example:

```text
before leaving: Track 7  -> P001
after returning: Track 26 -> P001
```

---

## 5. Gallery Rule

Do NOT automatically enroll every detected pedestrian.

Correct behavior:

```text
all people -> temporary tracks
user explicitly enrolls target -> persistent Person ID
```

Only user-enrolled people belong to the persistent Target Gallery.

---

## 6. Reference Repositories

The `references/` directory contains third-party projects for reading only.

Expected structure:

```text
references/
├── ultralytics/
├── deep-person-reid/
├── Re-id_initial/
├── person-reid-yolov8-tracking/
├── person_reid_yolo/
└── boxmot/
```

### Priority 1: actual implementation references

#### `references/ultralytics`
Use for:

- YOLO inference
- tracking API
- BoT-SORT
- `Results.boxes.id`
- tracker configuration
- camera/video input patterns

The application should normally depend on the installed `ultralytics` package rather than importing source files from `references/ultralytics`.

#### `references/deep-person-reid`
Use for:

- OSNet
- Person ReID preprocessing
- FeatureExtractor
- embedding normalization

### Priority 2: business-logic references

#### `references/Re-id_initial`
Study especially:

```text
gallery.py
reid_model.py
object_tracking_reid.py
```

Use it to understand a compact YOLO + tracker + OSNet + gallery pipeline.

#### `references/person-reid-yolov8-tracking`
Use mainly for:

- GlobalIdentityManager ideas
- multiple embeddings per identity
- centroid embeddings
- re-entry matching
- identity memory

Do NOT copy its Simple IoU Tracker design into the main application.
Do NOT reproduce the fixed 477-frame/demo assumptions.
Do NOT automatically create permanent IDs for all pedestrians.

#### `references/person_reid_yolo`
Use mainly for:

- SQLite persistence ideas
- loading identities across runs
- person database organization

Do not base the new detector implementation on its older YOLO code.

### Priority 3: fallback tracker reference

#### `references/boxmot`
Use if:

- Ultralytics BoT-SORT is not sufficient
- many ID switches occur
- another tracker such as DeepOCSORT/StrongSORT is needed
- tracker/ReID abstraction needs a mature reference

Do not introduce BoxMOT into runtime dependencies without a concrete reason.

---

## 7. Reference Directory Is Read-Only

Treat all files under `references/` as read-only third-party reference material.

Do not:

- implement project features directly inside `references/`
- edit reference repositories
- make the application depend on relative imports from reference source trees
- mix copied code from several projects into one large module

All project code belongs in the main project directories such as `src/`, `ui/`, and `tests/`.

---

## 8. Preferred Main Project Structure

```text
src/
├── models.py
├── tracking_pipeline.py
├── reid.py
├── gallery.py
├── gallery_recognition.py
├── target_manager.py
├── database.py
├── video_source.py
├── visualization.py
├── roi_selector.py
├── reid_frame_cache.py
└── utils.py

ui/
└── opencv_ui.py

tests/
├── test_gallery.py
├── test_database.py
├── test_roi_match.py
├── test_reid_similarity.py
└── test_target_manager.py
```

Avoid putting all logic in `app.py`.

---

## 9. Module Responsibilities

### `tracking_pipeline.py`

Owns detector/tracker integration only.

Public contract should remain roughly:

```python
process(frame) -> list[Track]
```

This abstraction must make it possible to replace Ultralytics tracking with BoxMOT later without rewriting business logic.

### `reid.py`

Owns OSNet loading, preprocessing, inference, and embedding normalization.

Preferred API:

```python
extract(crop)
extract_batch(crops)
```

### `gallery.py`

Owns persistent-person identity matching logic.

Must support:

- multiple embeddings per Person ID
- centroid
- search
- add/update/delete
- bounded embedding count

### `gallery_recognition.py`

Owns scheduled automatic matching of eligible current Tracks against the in-memory
Gallery. It performs batch ReID extraction, centroid similarity, threshold/margin
checks, confirmation, and one-to-one assignment; it must not own SQLite or enroll
new Gallery people.

### `reid_frame_cache.py`

Owns only same-frame embedding reuse between Recovery and Gallery recognition. A
different frame index must invalidate the cache.

### `target_manager.py`

Owns current selected target state:

```text
NONE
TRACKING
LOST
```

It should not own YOLO or database internals.

### `database.py`

Owns SQLite persistence only.

### `opencv_ui.py`

Owns mouse/keyboard interaction and display behavior.

---

## 10. Development Order

Implement incrementally.

Do not attempt the complete application in one large patch.

Required sequence:

```text
MVP-0 project skeleton/config/logging
MVP-1 YOLOv8n person detection + local video/camera input
MVP-2 BoT-SORT Track IDs
MVP-3 manual ROI target selection
MVP-4 OSNet embedding extraction
MVP-5 LOST -> ReID -> RECOVER
MVP-6 in-memory TargetGallery
MVP-7 SQLite persistence
MVP-8 automatic gallery recognition
MVP-8.1 crowded-scene and occlusion robustness
MVP-8.2 safe persistent Gallery feature enrichment
MVP-9 RTSP input and live-stream robustness
MVP-10 presentation/UI/performance polish
```

Each stage should remain runnable before moving to the next stage.

---

## 11. Coding Rules

1. Prefer small modules and explicit interfaces.
2. Use type hints for public APIs.
3. Prefer dataclasses for `Detection`, `Track`, and match results.
4. Keep all thresholds and paths in config files.
5. Use `logging`, not uncontrolled `print()` calls.
6. Do not hard-code video paths or frame counts.
7. Do not hard-code CUDA; support CPU fallback when practical.
8. Keep model loading outside the per-frame loop.
9. Avoid unnecessary ReID inference on every person every frame.
10. Batch ReID when convenient.
11. Normalize embeddings consistently before cosine matching.
12. Keep Gallery size bounded per identity.
13. Never update Gallery with a low-confidence or uncertain identity match.
14. Do not silently swallow exceptions in the main processing loop.
15. Preserve simple runnable behavior over premature optimization.

---

## 12. Configuration Rule

Do not scatter magic numbers in source code.

MVP-1 actively reads only the input, YOLO, runtime, and display settings it needs;
later-stage tracker, ReID, gallery, and storage settings may be reserved in the
configuration without being implemented early.

Configuration should cover at least:

```text
video source
YOLO weights
device
detection confidence
tracker backend
lost frame count
ReID threshold
target recovery threshold
max embeddings per person
embedding update interval
minimum person crop size
database path
thumbnail directory
UI flags
```

For MVP-1, configuration must include at least: video source, YOLO weight path,
person class ID, confidence/IoU/image-size settings, device selection, Windows
`num_workers=0`, and basic display options.

---

## 13. ReID Execution Policy

Do not run full Gallery ReID blindly for every track on every frame.

Highest priority:

- user-selected target
- LOST target recovery
- newly created track

Lower priority:

- periodic identity refresh
- periodic Gallery embedding update

This is important for real-time performance.

---

## 14. Gallery Safety Rules

When adding new embeddings to an existing person:

- prefer high-quality crops
- prefer stable track/person bindings
- enforce an update interval
- keep a maximum count
- do not append an embedding solely because one weak match barely crossed the threshold

If needed, use repeated confirmation before updating a permanent identity.

---

## 15. Testing Expectations

When implementing or changing business logic, add or update tests where practical.

Prioritize tests for:

- ROI-to-track IoU selection
- Gallery matching
- embedding centroid logic
- SQLite round trip
- Person ID generation
- TargetManager state transitions
- LOST/recovery logic with mocked embeddings

GPU-heavy integration tests can remain separate from normal unit tests.

---

## 16. Change Discipline

When asked to implement a feature:

1. Inspect the existing relevant modules first.
2. Inspect the most relevant reference project only if useful.
3. Make the smallest coherent change.
4. Keep existing interfaces stable where possible.
5. Run relevant tests or a minimal smoke test.
6. Report what changed and any unresolved runtime dependency/model-weight requirement.

Do not refactor unrelated parts of the project unless necessary.

---

## 17. When Architecture Changes Are Allowed

A major architecture change is acceptable only when there is a concrete reason such as:

- current API cannot support required functionality
- tracker quality is insufficient in test footage
- a dependency is unusable in the target environment
- measured performance requires a different backend

Before making such a change, explain the problem and proposed impact.

Examples:

```text
Ultralytics BoT-SORT -> BoxMOT DeepOCSORT
OpenCV UI -> PySide6
PyTorch -> ONNX/TensorRT
```

These are later-stage choices, not default first-version choices.

---

## 18. Definition of Done for Core Project

The core project is considered functionally complete when all of the following work in one integrated application:

- video/camera person detection
- stable temporary Track IDs
- manual target selection
- continuous highlight of selected target
- target LOST state
- target recovery after leaving and returning
- explicit user enrollment into Target Gallery
- persistent SQLite storage
- automatic recognition of enrolled people after restart
- ordinary pedestrians are not automatically enrolled
- basic logs and config files exist
- critical Gallery/TargetManager logic has tests

Presentation polish is secondary to these requirements.
