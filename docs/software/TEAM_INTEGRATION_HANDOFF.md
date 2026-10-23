# Team integration handoff: exported metric depth

## Scope and current status

Yarema runs pretrained inference on his own computer. Marko imports already
computed arrays and evaluates the existing reactive controller through replay
and the mock drone. No neural dependencies, model downloads or real model
execution are required on Marko's computer for this workflow.

The controller, sector statistics, hysteresis/braking, stale-depth checks,
commands, video reader, offline/virtual/realtime scheduling, GUI rendering,
CSV/JSON logging and guarded/mock drone adapters were already implemented.
The missing component was a portable, validated exchange artifact: the legacy
index-only NPY loader could not establish camera/units/frame provenance.

Added now: a versioned JSON + float32 NPY exporter/importer, explicit calibration
and identity checks, CLI selection, per-frame provenance/latency logs, synthetic
handoff fixtures and 23 focused exchange tests. The controller, teammate wrappers
and inference preprocessing/scaling are unchanged. The original 64 tests remain
intact; the expanded suite has **87 passing tests**.

## 1. What Yarema must provide

For each recorded scenario, provide:

1. The **unchanged original video**, and synchronized frame timestamp CSV if
   available. Export with the same reader/decode order used for replay.
2. One prediction bundle per model/checkpoint, with one aligned output per
   decoded frame, including frames where output is wholly invalid. Export those
   invalid arrays explicitly rather than omitting a prediction.
3. The camera calibration file actually used: JSON `size: [width,height]`, `K`
   3x3 matrix and finite one-dimensional `dist`. Include its identifier, camera
   identity, geometry and verification status.
4. Exact model name/kind and checkpoint ID plus revision/hash identifier. Do not
   substitute a guessed checkpoint name or invent camera parameters.
5. Per-frame original-camera K and scalar focal passed into the existing wrapper,
   actual preprocessing/restoration details, and optional measured inference
   latency. If unavailable, latency is explicitly `null`; required provenance
   cannot be omitted or filled with guessed values.

The iPhone-labeled calibration in `results/calib_iphone_960.json` is not Tello
calibration. Device identity/calibration verification is a team responsibility;
matching array dimensions alone does not establish correct physical scale.

## 2. Version-1 exchange format

One bundle holds exactly one scenario/model/checkpoint. All data paths inside
it are relative, confined to the bundle; absolute paths, parent traversal and
escaping symlinks are rejected. Example generated layout:

```text
outputs/prediction_handoff/
  video.avi
  timestamps.csv
  synthetic_calibration.json
  scenarios.json                 # multi-run replay manifest
  da3/
    predictions.json             # exchange manifest
    calibration.json             # byte-identical calibration copy
    depth/000000.npy
    depth/000001.npy
    ...
  metric3d/
    predictions.json
    calibration.json
    depth/000000.npy
    ...
  invalid/
    missing_prediction/predictions.json
    timestamp_mismatch/predictions.json
    calibration_mismatch/predictions.json
    corrupted_file/predictions.json
    radial_depth/predictions.json
```

Every array is native NumPy **float32 HxW**, aligned to the original decoded RGB.
The exporter does not cast float64, fill holes, resize, normalize values or apply
metric scaling. Unsupported dtype or shape fails clearly. `.npy` loading disables
pickle. NaN, positive/negative infinity, zero and negative values are saved
unchanged; they are unknown/invalid, never evidence of free space.

Top-level required metadata:

| Field | Meaning / validation |
| --- | --- |
| `schema_version` | Integer 1; other versions rejected |
| `scenario_id` | Nonempty shared scenario identity |
| `frame_count` | Positive count of all decoded source frames |
| `video_sha256` | Original video file identity; do not re-encode after export |
| `model.name` | Actual model name; synthetic fixtures explicitly say SYNTHETIC |
| `model.kind` | `da3` or `metric3d` |
| `model.checkpoint_id`, `checkpoint_revision` | Both required; supply real checkpoint/revision identification, or explicitly synthetic identifiers for fixtures |
| `model.origin` | `pretrained` or `synthetic`; not inferred from the name |
| `calibration.id`, `camera_id` | Nonempty identities that Marko must explicitly match |
| `calibration.file`, `sha256` | Relative calibration copy and its hash |
| `calibration.status` | `verified`, `provisional` or `synthetic` |
| `calibration.geometry` | `distorted` or `undistorted`; undistorted requires corresponding zero-distortion calibration |
| `depth.dtype` | `float32` |
| `depth.units` | Required; replay accepts only `m` |
| `depth.convention` | Required; replay accepts only `optical_axis_z` |
| `depth.invalid_values` | Exactly `nonfinite_or_nonpositive` |
| `depth.saturation_m` | Explicit `null` or positive saturation threshold; current Metric3D wrapper requires `300.0` |
| `frames` | Ordered complete frame records; no duplicate indices/files or gaps |

Each `frames` record requires:

| Field | Meaning / validation |
| --- | --- |
| `frame_id` | Canonical contract ID `<scenario_id>:<zero-padded-6-digit-index>`; same ID across model bundles |
| `frame_index` | Zero-based decoded video index; full sequence 0..frame_count-1 |
| `timestamp_s` | Original finite nonnegative timestamp, strictly increasing |
| `timestamp_source` | `external`, `fps_approximation`, or `synthetic_exact`; retained without upgrading its precision |
| `image_shape`, `depth_shape` | Both `[height,width]`, equal; RGB itself must be uint8 HxWx3 |
| `calibration_id` | Must match bundle calibration identity |
| `camera_intrinsics` | Original RGB camera's K at this frame size, matching uniform resize of calibration K |
| `inference_focal_px` | Actual positive scalar focal argument passed to `model.predict` |
| `preprocessing` | JSON object; required `input_color: RGB`, `output_alignment: original_rgb`, and nonempty `description`; preserve actual resize/pad/normalization/scaling/restoration details as additional fields |
| `rgb_sha256` | SHA-256 of contiguous decoded RGB bytes; verifies exact frame content |
| `depth_file`, `depth_sha256` | Relative NPY filename and file integrity hash |
| `inference_latency_s` | Nonnegative measured inference time or explicit `null` |

Scenario/model/calibration/depth metadata is stored once at bundle level and
applies to every prediction. Per-frame metadata is stored separately. JSON
contains no NaN/Infinity literals, duplicated keys or pickled objects; invalid
pixel values are represented in NPY arrays only. Unknown optional JSON fields
are preserved by the exporter, allowing actual checkpoint/software/preprocessing
provenance to be included without redesigning version 1.

Frame IDs in this version are assigned canonically by the format. If another
capture system uses different IDs, keep them as additional metadata rather than
replacing this canonical ID without a versioned reader change.

## 3. How to export predictions

The reusable implementation is `tello_nav/replay/predictions.py`:

- `PredictionExporter(...)` creates a **new** output directory, refusing an
  existing path. It validates required metadata and copies the calibration.
- `add(frame, depth_map, ...)` saves the original float32 array with exclusive
  file creation, validates metadata, and associates it with one canonical frame.
- `finalize()` requires the full contiguous sequence before creating
  `predictions.json`. Duplicate/out-of-order frames, missing predictions and
  repeated finalize calls fail. Failed/incomplete jobs must use a fresh output
  directory; they are not silently resumed or overwritten.

The following is a **call-site example for Yarema's inference script**, not a
script run on this machine. Variables must come from his actual configuration,
checkpoint and capture metadata. Do not replace missing values with guesses.
The existing model call is unchanged:

```python
from tello_nav.replay.predictions import PredictionExporter

exporter = PredictionExporter(
    output_directory,
    scenario_id=scenario_id,
    frame_count=decoded_frame_count,
    video_path=video_path,
    model={
        "kind": model_kind,                 # "da3" or "metric3d"
        "name": model.name,
        "checkpoint_id": checkpoint_id,
        "checkpoint_revision": checkpoint_revision,
        "origin": "pretrained",
    },
    calibration_path=calibration_path,
    calibration_id=calibration_id,
    camera_id=camera_id,
    calibration_status=calibration_status,  # verified only when established
    calibration_geometry=calibration_geometry,
    depth={
        "dtype": "float32",
        "units": "m",
        "convention": "optical_axis_z",
        "invalid_values": "nonfinite_or_nonpositive",
        "saturation_m": 300.0 if model_kind == "metric3d" else None,
    },
)

# Inside Yarema's existing frame loop, after choosing true frame metadata:
rgb = frame.rgb
# Existing inference; performed on Yarema's computer only:
depth_map = model.predict(rgb, fx)
exporter.add(
    frame,
    depth_map,
    camera_intrinsics=K_for_original_rgb.tolist(),
    inference_focal_px=float(fx),
    preprocessing=actual_preprocessing_metadata,
    inference_latency_s=measured_inference_latency_or_none,
)

# After all source frames have been processed:
manifest_path = exporter.finalize()
```

Use `VideoReader(video_path, timestamps_path)` to share decode order and original
timestamps. Frame count must match the full decoded video. If a model fails,
Yarema can explicitly export an all-NaN float32 HxW map and record the failure
in extra preprocessing/frame provenance; do not fabricate predictions or omit
that frame. If a model already returns float32, no dtype operation is needed.
If it returns another dtype, decide on and document an explicit float32 conversion
upstream; the exporter will not silently do it.

For known wrappers, preprocessing descriptions should include actual settings:
DA3 checkpoint, internal resize resolution/method, focal/300 scaling and output
restoration; Metric3D's 616x1064 resize/pad, mean/std, canonical focal conversion,
output restoration and 300m clamp. These details must describe the actual run,
not be copied as assertions before verifying the installed model code. The
exporter never loads a checkpoint or performs any of these transformations.

## 4. Import checks and metric-evaluation eligibility

`ManifestDepthProvider` accepts a bundle plus expected scenario, video,
calibration path, calibration ID and camera ID. The expected calibration should
be an independently agreed copy, not accepted as verified merely because a
bundle contains it. It checks:

1. Schema, required metadata, identifiers, units/convention and provenance.
2. Original video and calibration SHA-256 identity; complete unique ordered
   frame records; calibration geometry and per-frame K.
3. One full source-video decode during initialization to verify frame count.
   Playback then uses the existing reader normally. This adds a startup pass but
   cannot mistake omitted trailing predictions for a complete export.
4. At every prediction request: frame ID/index, original timestamp within
   **1 microsecond absolute tolerance**, timestamp source, RGB geometry/content
   hash, depth file/hash, dtype and shape. No nearest-timestamp matching occurs.

Different decoders/library versions can produce slightly different RGB pixels
from the same compressed video. The strict RGB hash then rejects the mismatch;
coordinate the same decode pipeline or re-export against the agreed decoded
frames. There is intentionally no silent tolerance that could conceal a wrong
frame. Export/replay original timestamps in the same epoch; the runner alone
normalizes them for scheduling and retains originals in logs.

Structural/semantic/calibration failures reject the bundle before controller
execution. Frame-level missing/corrupt files or alignment failures raise a clear
error; the existing runner records it, passes missing depth to the controller,
and produces STOP with normal recovery rules. A log can therefore exist despite
per-frame failures: inspect `error`, not just process exit or file existence.

Eligibility rules:

- Real (`origin=pretrained`) predictions require calibration status `verified`,
  matching expected IDs/hash, meters and optical-axis Z.
- `provisional` calibration is stored but **cannot be used by this strict
  controller-evaluation import path**. There is no provisional override flag.
- Synthetic prediction/calibration pairs require
  `--allow-synthetic-predictions`. Logs mark them `synthetic: true` and
  `metric_evaluation_eligible_by_metadata: false`; GUI source labels show
  `synthetic`. Their names/checkpoint IDs explicitly identify fixtures.
- `verified` is the export owner's assertion, not a new physical calibration
  certificate. Passing hash/schema checks does not measure model accuracy or
  prove that intrinsics belong to a Tello. Real eligibility by metadata must not
  be presented as independently validated metric accuracy.

The depth files stay unchanged. On import, declared saturation is masked as
unknown on an in-memory copy. This prevents Metric3D's clamp from making an
infinity failure look like a clear 300m observation. Remaining invalid pixels
are ignored by sector statistics; insufficient valid coverage causes STOP.
Small holes below configured coverage thresholds remain accepted uncertainty.

Millimeter, radial-range, relative, inverse-depth or cropped outputs are not
silently repaired. Perform any justified conversion explicitly before exporting
an aligned metric artifact and record the transformation. Millimeters to meters
uses /1000; pinhole undistorted ray range to Z requires division by
`norm(inv(K) @ [u,v,1])` at each pixel. Do not apply that formula to distorted
pixels without appropriate camera geometry. No empirical scale factor is invented.

The legacy `--depth-dir` path remains for backwards-compatible array development.
Its new metadata explicitly says it is not eligible for metric evaluation by
metadata. It has no equivalent identity/calibration guarantee; use `--predictions`
for teammate handoff and do not claim raw-file mode validates provenance.

## 5. Working offline demonstration and model selection

All commands below run from the repository root with existing NumPy/OpenCV.
No model package, Torch, GPU, SDK, network or weights are needed for exchange
replay. Optional wrapper-body tests in the full suite use CPU Torch if present.

Generate fresh fixtures (the command deliberately refuses an existing destination):

```sh
python -B -m examples.create_prediction_fixture --output outputs/prediction_handoff
```

If that path was generated earlier, reuse it or choose a new empty output path;
do not delete existing work just to rerun the generator.

The supplied replay manifest runs both model-labeled synthetic bundles against
the same eight recorded fixture frames:

```sh
python -B -m tello_nav --manifest outputs/prediction_handoff/scenarios.json --allow-synthetic-predictions --mock-drone --output outputs/prediction_handoff/replay.json
```

Select only the DA3-labeled bundle:

```sh
python -B -m tello_nav --video outputs/prediction_handoff/video.avi --timestamps outputs/prediction_handoff/timestamps.csv --predictions outputs/prediction_handoff/da3/predictions.json --scenario-id exchange-demo --calibration outputs/prediction_handoff/synthetic_calibration.json --calibration-id synthetic-camera-v1 --camera-id synthetic-camera --allow-synthetic-predictions --mock-drone --output outputs/prediction_handoff/da3-replay.json
```

Switch to the Metric3D-labeled bundle without changing controller code:

```sh
python -B -m tello_nav --video outputs/prediction_handoff/video.avi --timestamps outputs/prediction_handoff/timestamps.csv --predictions outputs/prediction_handoff/metric3d/predictions.json --scenario-id exchange-demo --calibration outputs/prediction_handoff/synthetic_calibration.json --calibration-id synthetic-camera-v1 --camera-id synthetic-camera --allow-synthetic-predictions --mock-drone --output outputs/prediction_handoff/metric3d-replay.csv
```

The replay-manifest `id` is a unique run/log label. Optional `scenario_id` is the
source bundle's shared scenario identity; it defaults to `id`. This allows two
model runs for the same source scenario without duplicate run IDs. Original
scenario/frame identities remain in provider metadata and each record's
`prediction.source_scenario_id` / `frame_id`.

For terminal stepping, add `--step` to a single-run offline/virtual command.
For GUI RGB/depth/sector/controller display, add `--visualize` on a display-capable
host. Realtime stepping is intentionally rejected. To exercise controlled delays:

```sh
python -B -m tello_nav --manifest outputs/prediction_handoff/scenarios.json --allow-synthetic-predictions --mock-drone --mode virtual --inference-delay .4 --communication-delay .2 --late-policy drop --max-lateness .2 --output outputs/prediction_handoff/delayed.json
```

`--mode realtime` uses actual decode/load/processing time plus configured delay;
`offline` and `virtual` decisions are deterministic for identical artifacts,
configuration and decoding. Measured execution time is observational and differs
between runs. `prediction.exported_inference_latency_s` is the original measured
network time, not file-loading time. The legacy `measured_inference_s` field now
measures provider/file-loading time in this path. Artificial delays remain
explicit configuration; exported network latency is not silently applied twice.

The mock adapter's independent .3s freshness/watchdog bound can reject a command
before the controller's .5s bound. Such inhibition latches, and logs preserve the
requested controller command separately from the effective mock STOP command.

### Deliberately invalid fixtures

The `invalid/` bundles are diagnosis examples. Use the DA3 single-run command
above but replace the `--predictions` argument with the case path:

| Case | Expected behavior |
| --- | --- |
| `invalid/missing_prediction/predictions.json` | Reject bundle for missing prediction |
| `invalid/timestamp_mismatch/predictions.json` | Frame 0 error/STOP, no use of mismatched prediction |
| `invalid/calibration_mismatch/predictions.json` | Reject expected calibration identifier mismatch |
| `invalid/corrupted_file/predictions.json` | Frame 0 hash error/STOP |
| `invalid/radial_depth/predictions.json` | Reject unsuitable depth convention |

Use a new output log path when investigating each case. Structural failure does
not produce a completed output log; a frame error may produce a log with STOP.
Existing log files are not deleted on failure, so do not confuse an older log
with a new successful run.

## 6. Replay with Yarema's real exported predictions

The existing CLI is sufficient; no neural dependency installation is needed.
Set actual local paths/identifiers for the received video, timestamp CSV,
agreed calibration and one model's exported manifest. No fictitious real data
filename is supplied here:

```sh
: "${REPLAY_VIDEO:?Set actual unchanged video path}"
: "${FRAME_TIMESTAMPS:?Set synchronized frame timestamp CSV path}"
: "${PREDICTIONS:?Set one model bundle predictions.json path}"
: "${CAMERA_CALIBRATION:?Set agreed calibration JSON path}"
: "${CALIBRATION_ID:?Set agreed calibration identifier}"
: "${CAMERA_ID:?Set agreed physical camera identifier}"
: "${SCENARIO_ID:?Set exported scenario identifier}"

python -B -m tello_nav --video "$REPLAY_VIDEO" --timestamps "$FRAME_TIMESTAMPS" --predictions "$PREDICTIONS" --scenario-id "$SCENARIO_ID" --calibration "$CAMERA_CALIBRATION" --calibration-id "$CALIBRATION_ID" --camera-id "$CAMERA_ID" --mock-drone --output outputs/imported-real-replay.json
```

Do **not** add `--depth-model` or `--allow-model-download`: those select local
model execution, not external predictions. Do not use the synthetic override
for real data. To switch between real DA3 and Metric3D outputs, select the other
bundle with the same source video/scenario/camera expectations. To compare both
in a single run, use unique `id` labels and shared `scenario_id` in a replay
manifest, following the generated `scenarios.json` structure. Its file paths are
relative to the replay manifest, while each bundle's depth/calibration paths are
relative to that bundle's `predictions.json`.

If no synchronized CSV exists, omit `--timestamps` only when Yarema also used
the identical FPS-derived timestamps. Both sides must retain `fps_approximation`;
that source must never be promoted to exact capture timing.

## 7. Inspecting decisions and provenance

```sh
python -m json.tool outputs/prediction_handoff/replay.json
```

JSON has metadata and records. CSV stores nested dictionaries as JSON cells
plus a `.metadata.json` sidecar. Inspect:

- Scenario/run ID, source scenario/frame IDs, original timestamp and timestamp
  source, model/checkpoint and calibration metadata.
- Sector percentiles/minima, valid counts/coverage, controller state, four RC
  channels, reason, error and safety-event fields.
- `mock_execution.accepted`, `inhibited`, `effective_command` and `error`, rather
  than assuming a requested command was accepted.
- Imported-versus-simulated timing fields and source origin/eligibility flags.

Hashes establish consistency and accidental-corruption detection, not signatures
or independent trust in metadata. The file checks cannot establish actual model
weights, real camera identity or physically accurate scale by themselves.

## 8. Tests, packaging and verified results

```sh
python -B -m unittest discover -s tests -v
python -B -m pip wheel --no-build-isolation --no-deps --no-cache-dir --wheel-dir /tmp/tello-nav-handoff-wheel .
```

All **87 tests passed**, including the original 64 and 23 new exchange tests.
New coverage includes exact float32/invalid-value preservation, no overwrite,
incomplete export, identity/timestamp/K/units checks, missing/corrupt NPY,
malformed/duplicate metadata, saturation, path confinement, model switching,
deterministic import, lazy stepping, delayed STOP and mock execution. Existing
coverage supplies free/left/right/center/blocked/thin cases, hysteresis, stale
observations, command bounds, operator STOP and out-of-order commands. No
pretrained inference or physical drone was used by these tests.

The generated fixture round trip produced **16 decisions, all 16 accepted by the
mock boundary**. DA3-labeled synthetic: STOP=3, FORWARD=1, BRAKE=4.
Metric3D-labeled synthetic: STOP=4, FORWARD=1, BRAKE=3; saturation causes additional
unknown coverage/STOP. These are software-fixture outcomes, not model scores or
flight results. No data is labeled as genuine network output.

Code/package files changed this turn:

- New `tello_nav/replay/predictions.py`.
- New `examples/create_prediction_fixture.py` and `tests/test_prediction_exchange.py`.
- Updated `tello_nav/cli.py` to select exchange bundles and expected identities.
- Updated `tello_nav/replay/runner.py` for per-frame provenance logging.
- Updated `tello_nav/replay/providers.py` to label legacy NPY provenance limitations.
- Updated `tello_nav/replay/visualization.py` to display prediction origin.
- Added this handoff and linked it from the existing software guide.

## 9. Remaining verification responsibilities

**Yarema:** execute actual pretrained networks on his computer; identify real
checkpoint/code versions; verify preprocessing/restoration, invalid/confidence
masks and metric Z scaling; export aligned full-frame predictions and real
metadata; validate numerical outputs against known scene distances.

**Camera/data team:** supply actual recorded RGB/timestamps and correctly matched
Tello intrinsics/geometry, establish physical calibration validity and measured
reference distances. Provisional/iPhone parameters are not Tello calibration.

**Future live integration:** verify actual sensor freshness, reliable telemetry,
measured braking, operator procedures, Wi-Fi failure behavior and hardware adapter
operation under separate authorization. No physical operations occur in this task.

Replay continues through the same fixed recorded frames regardless of commands.
It **does not simulate alternative drone trajectories or prove collision
avoidance**. Monocular predictions, even correctly packaged and fully finite,
do not guarantee obstacle-free movement. Flight experiments and poster remain
deferred. No commits, pushes or branches were created.
