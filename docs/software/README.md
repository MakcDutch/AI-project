# Controller, Replay Stand, and Tello integration

## Repository inspection and scope

Before implementation, the repository contained only `README.md`, the Ukrainian
proposal `docs/proposal/main.tex`, and its bibliography. There were no model
implementations, calibration files, recordings, Python dependencies, controller,
or drone SDK integration. The existing files were preserved.

This package implements Marko-Zenon Peleshko's software responsibilities. It does
not implement flight experiments, model evaluation metrics, calibration, dataset
collection, or poster generation. No model weights are downloaded.

## Setup and quick start

Python 3.10+ is required. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m tello_nav --synthetic empty_corridor --output outputs/corridor.json
python -m tello_nav --synthetic center_obstacle --controller-config examples/controller.json --output outputs/obstacle.csv
python -m tello_nav --synthetic invalid_depth --output outputs/invalid.json
```

The only required libraries are NumPy and OpenCV. Tests use the standard library
`unittest`, NumPy, and an OpenCV MJPG encoder/decoder for small temporary videos;
no pytest, GPU, model weights, network, or physical drone is required. For a
server without GUI libraries, `opencv-python-headless` can supply `cv2` instead
of `opencv-python` (install only one OpenCV distribution); omit `--visualize`.
The installed `tello-replay` command and `python -m tello_nav` are equivalent.

## Architecture

```text
VideoReader / synthetic_frames
    -> Frame(index, timestamp, RGB, timestamp_source)
    -> DepthProvider.predict(frame) -> aligned HxW camera-Z meters
    -> ReactiveController.compute(depth_map, timestamp, telemetry)
    -> ControlCommand + sector diagnostics
    -> JSON/CSV logger + optional OpenCV viewer

The same controller -> MockDroneAdapter (default)
                    -> TelloAdapter (explicit live selection + operator authorization)
```

`tello_nav/controller` has no OpenCV or drone SDK dependency. `replay` never
imports the live adapter and has no option that transmits commands. `drone`
validates commands at the hardware boundary independently of the controller.
A new controller instance is created for each replay scenario/run.

## Depth contract and model integration

`DepthProvider` is a Python protocol in `tello_nav/replay/providers.py`:

```python
class DepthProvider(Protocol):
    name: str
    def predict(self, frame: Frame) -> np.ndarray: ...
```

The output must be a real numeric `(H, W)` array aligned with the input RGB
pixels. Distances are **camera optical-axis Z in meters**, not inverse depth,
relative depth, or Euclidean ray range. NaN, infinity, zero and negative values
are invalid, never free space. The provider does not fill holes or resize a
mismatched result. The controller ignores invalid pixels for statistics and
stops if any sector has inadequate coverage. Small holes below that threshold
remain an explicitly accepted uncertainty; increase `min_coverage` to 1 to
require complete observations in the ROI.

Neither Depth Anything 3 nor Metric3D is implemented in this repository.
There is deliberately no fabricated model adapter or scaling factor. Both
can connect through `CallableDepthProvider` when Yarema supplies a validated
metric inference function:

```python
from tello_nav.replay.providers import CallableDepthProvider
from tello_nav.replay.runner import ReplayRunner
from tello_nav.replay.video import VideoReader

# team_infer(frame) must already meet the full metric/alignment contract.
provider = CallableDepthProvider("metric3d-<actual-variant>", team_infer)
rows = ReplayRunner(provider).run(VideoReader("data/corridor.avi"), "corridor")
# Switch to Depth Anything 3 by supplying its validated function and variant name.
```

This is a programmatic inference integration point; there is no working
`--model metric3d` or `--model da3` flag. Before connecting either model, document
its exact checkpoint/variant, public API, input color/normalization, resize/pad
and crop transformations, camera intrinsics (including transformed focal length),
output units, scale treatment, Z versus ray-depth convention, and invalid/confidence
mask. Validate absolute scale on the team's measured data. Relative-only output
is unsuitable; naming a provider "metric3d" does not make its output metric.

`MockDepthProvider` implements eight deterministic depth fixtures, including
unknown reflective-surface and invalid-depth scenarios. Those fixtures do not
predict how a real network behaves on glass, reflections, or thin obstacles.

## Controller behavior and configuration

Load a JSON configuration with `--controller-config examples/controller.json`.
All `ControllerConfig` and nested `SectorConfig` fields are accepted; unknown
fields and invalid thresholds fail immediately.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_distance_m` | 1.0 | Base obstacle threshold in camera-Z meters |
| `hysteresis_m` | 0.25 | Additional clearance required to recover/choose a side |
| `max_frame_age_s` | 0.5 | Maximum observation age at decision time |
| `brake_duration_s` | 0.3 | Minimum zero-translation dwell before side selection |
| `avoid_hold_s` | 0.5 | Minimum direction hold while selected side stays clear |
| `recovery_frames` | 2 | Consecutive clear observations before forward recovery |
| `command_limit` | 20 | Absolute RC channel limit (SDK range is at most 100) |
| `forward_speed` / `lateral_speed` | 15 / 10 | RC setpoints; not measured m/s |
| `allow_lateral` | false | Explicit opt-in to lateral decisions |
| `sectors.roi` | `[0,.2,1,.85]` | Normalized x0,y0,x1,y1 crop |
| `sectors.boundaries` | `[1/3,2/3]` | Horizontal sector boundaries within crop |
| `sectors.percentile` | 10 | Lower-percentile proximity statistic |
| `sectors.min_coverage` | .9 | Required valid fraction **in every sector** |
| `sectors.min_valid_pixels` | 16 | Required absolute valid pixel count per sector |

Each sector reports percentile distance, minimum distance, valid pixel fraction,
valid count, total count and known/unknown status. An otherwise clear percentile
is overridden by **any** close valid pixel: this catches thin obstacles smaller
than the percentile support, at the expense of false stops on outliers. This
still cannot detect obstacles that the model itself misses.

State transitions:

- Initial STOP requires `recovery_frames` observations with center depth above
  `min_distance_m + hysteresis_m` before FORWARD.
- FORWARD continues while center depth stays above the base threshold. A close
  center pixel commands BRAKE immediately; every BRAKE channel is zero.
- BRAKE waits its dwell. A recovered center can resume FORWARD. A blocked center
  can select a sufficiently clear left/right sector only when `allow_lateral`
  is true; otherwise STOP. Equal distances choose left deterministically.
- AVOID_LEFT/AVOID_RIGHT command lateral translation only. They keep their
  chosen direction while clear; a newly unsafe chosen side immediately enters
  BRAKE before another side can be selected. Center recovery also enters BRAKE
  before returning to FORWARD. There is no direct left-to-right transition.
- Missing maps, inference failures, invalid shapes, insufficient coverage,
  nonfinite/stale/future/repeated or backward timestamps immediately cause STOP.
  Recovery again requires consecutive clear observations. Known but blocked
  input can re-enter BRAKE to reconsider an escape route.

`allow_lateral: true` is useful for inspecting avoidance decisions in replay.
A forward monocular image does not establish clearance for the drone body to
move sideways, or show objects outside the ROI/FOV. The default therefore
keeps lateral commands disabled. A zero RC request is not proof that physical
motion has stopped. No state here certifies obstacle-free movement.

For live or standalone use, inject a monotonic clock or use the default
`time.monotonic`. Observation timestamps and the clock must share the same
epoch and unit. Replay injects a recording-relative clock instead.

Optional `braking_deceleration_mps2` enables a larger safe distance only for
telemetry with `reliable=True`, finite nonnegative `speed_mps`, and a fresh
`timestamp` in the controller's clock domain:

```text
safe_distance = min_distance + speed * (observation_age + reaction_latency_s)
                            + speed^2 / (2 * braking_deceleration_mps2)
```

Telemetry older than `telemetry_max_age_s` (default .25) or missing/untrusted
telemetry uses the base distance. Raw Tello telemetry is deliberately marked
unreliable for this calculation. RC setpoints are never converted to speed.
Braking calibration and sufficiently conservative thresholds are team integration
requirements, not established facts about this aircraft.

## Recorded videos and precomputed depth

Paths below are examples for recordings supplied by the team; none are bundled:

```sh
python -m tello_nav --video data/corridor.avi --depth-dir data/corridor_depth --model-name metric3d-validated-variant --scenario-id corridor --output outputs/corridor.json
python -m tello_nav --video data/center.avi --timestamps data/center_times.csv --depth-dir data/center_depth --scenario-id center --output outputs/center.csv
python -m tello_nav --video data/corridor.avi --mock-depth center_obstacle --output outputs/synthetic_depth_on_video.json
```

A depth directory maps decoded frame indices exactly:

```text
corridor_depth/000000.npy  # decoded frame 0
corridor_depth/000001.npy  # decoded frame 1
...
```

Missing files and wrong shapes are logged as inference errors with STOP; the
loader never shifts another frame's depth into the missing slot. Files must
already follow the metric-depth contract; there is no automatic scale inference.
This mapping cannot detect a correctly named but semantically misaligned map:
the dataset producer must preserve the same video decode order, crop, and
resolution. Extra unused `.npy` files are not consumed. All `.npy` files in the
directory are fingerprinted in metadata.

Optional synchronized timestamp CSV:

```csv
frame_index,timestamp_s
0,0.000
1,0.104
2,0.203
```

Indices must be contiguous from zero and timestamps nonnegative, finite and
strictly increasing. Timestamp count must match decoded frames. Nonzero starting
timestamps are allowed and are normalized for scheduling; original values stay
in logs. Without CSV, `index / FPS` is labeled `fps_approximation`, not an exact
capture measurement. Invalid FPS requires an external CSV. Empty/unreadable
videos, malformed CSV, and frame-count/decode mismatches fail visibly. Some
codecs cannot distinguish truncated end-of-stream from legitimate EOF when
neither an external count nor a trustworthy advertised frame count exists.
The external timestamp producer is responsible for synchronization accuracy.

Multiple recordings use a JSON manifest, with paths relative to the manifest:

```json
[
  {"id": "corridor", "video": "corridor.avi", "depth_dir": "corridor_depth", "model_name": "metric3d-validated-variant"},
  {"id": "center", "video": "center.avi", "timestamps": "center_times.csv", "depth_dir": "center_depth", "model_name": "da3-validated-variant"}
]
```

```sh
python -m tello_nav --manifest data/scenarios.json --output outputs/scenarios.json
```

Each entry needs a unique `id`, `video`, optional `timestamps`, and exactly one
of `depth_dir` or `mock_depth`. `model_name` labels provenance, not a model loader.

## Timing, latency, and step-by-step processing

```sh
python -m tello_nav --synthetic center_obstacle --step --output outputs/steps.json
python -m tello_nav --synthetic empty_corridor --mode virtual --inference-delay .35 --communication-delay .2 --output outputs/delayed.json
python -m tello_nav --video data/center.avi --timestamps data/center_times.csv --depth-dir data/center_depth --mode realtime --late-policy drop --max-lateness .2 --output outputs/realtime.json
```

- **offline** processes all frames in order. Controller time follows source
  timestamps, independent of machine performance. Actual inference runtime is
  measured and logged but does not affect decisions. Artificial delays require
  virtual or realtime mode.
- **virtual** uses a deterministic single-worker queue. Start time is the later
  of frame arrival and previous completion; configured inference and communication
  delays advance virtual time. Actual Python/model runtime remains observational
  and is not added to simulated time. This deliberately allows repeatable latency
  sweeps across different computers.
- **realtime** paces arrival using source timestamps (or the explicitly approximate
  FPS timeline). Actual work plus artificial sleep delays contribute to age.
  Frames are processed sequentially with no catch-up speedup of inference.
- **late-policy process** handles late frames and lets freshness checks STOP.
  **drop** skips inference for frames already older than `max_lateness_s` at
  service start and logs a zero command plus a drop event. Frames becoming stale
  during inference are also stopped. No hidden frame dropping occurs in the reader.
- Communication delay is modeled before the final controller safety decision,
  representing a freshness check at delivery. There is no actuator dynamics,
  command transport simulator, or physical drone in replay.
- `ReplayRunner.iter_records` is lazy and yields one decision per `next()`.
  CLI `--step` pauses before fetching each next decision and is incompatible with
  realtime mode. CLI `--frames` controls synthetic sequence length only.

Offline/virtual **decisions** reproduce for identical observations, provider
outputs, timing/configuration and library versions. `measured_inference_s`
varies with runtime and is deliberately excluded from byte-for-byte determinism.
Learned providers must control their own seeds and nondeterministic operations.

## Logging and visualization

JSON contains `metadata` and `records`. CSV has one decision per row, nested
sector/command objects encoded as JSON cells, plus a `.metadata.json` sidecar.
Both include scenario ID, original frame timestamp/index and timestamp source,
model label, sector statistics and coverage, threshold, state, four RC channels,
reason, actual inference runtime, separate simulated inference/communication
delays, frame age, decision time, drops, errors and safety events.

CLI metadata includes all controller/replay configuration, input SHA-256 hashes,
source paths, synthetic scenario/length, and Python/NumPy/OpenCV versions.
Custom Python callers can use `runner.metadata()` and `write_results()` and
should add input/checkpoint/inference-code provenance themselves. Log files
are written on successful completion; input-validation failure interrupts the
run instead of exporting a misleading complete result. Output paths are user
selected and existing output files are replaced.

Add `--visualize` for RGB and depth panels, ROI/sector boundaries, sector
distances and coverage, state, command, timestamp and frame age. Depth colors
clip at 10 meters; black pixels are invalid, not free. Press q or Escape to end
that scenario's visualization. Headless execution is the default; display needs
a working OpenCV GUI environment. `--step` works in a terminal without a GUI.

## Drone adapter and operator authority

`create_adapter()` returns `MockDroneAdapter`. It stores commands locally and
cannot import the SDK or send network packets. The live SDK is an optional
extra (`python -m pip install -e '.[live]'`) and is imported only on an explicitly
authorized `TelloAdapter.connect()` call. Construction requires **both**
`mode="live"` and `operator_authorized=True`; application code must obtain that
operator decision, never infer it from a replay run. There is deliberately no
live-autonomy CLI or takeoff/landing API in this implementation.

The adapter exposes connect, start_video/video_frame (live only), telemetry,
validated send, emergency_stop, manual_override, explicit operator reset, watchdog,
and close. Commands require both a fresh command-creation timestamp and the
original observation timestamp. Send rechecks all four RC channel bounds and
STOP/BRAKE zero-motion rules. The default adapter limit is 20 and freshness
limit .3 seconds. Controller and adapter limits can be configured independently;
the stricter boundary wins.

A background live watchdog checks freshness independently of inference, and
latches STOP on timeout. Send failures latch inhibition and attempt zero RC;
the watchdog retries zero while inhibited. No automatic reconnection re-enables
motion. `manual_override()` revokes this adapter's motion authority and requests
zero RC, so a separate piloting interface can take over. It does not implement
keyboard piloting. `reset_operator_stop(operator_authorized=True)` is required
to restore this adapter's authority after inhibition.

**Emergency STOP here means a latched zero RC request, not immediate motor
cutoff, guaranteed hover, or landing.** UDP RC commands are unacknowledged and
may be lost; a severed Wi-Fi link cannot be made safe by a software watchdog.
The operator needs an independent aircraft emergency procedure. No automatic
takeoff, movement or landing occurs on connect. Closing requests zero and stops
video; it intentionally avoids `Tello.end()`, whose SDK cleanup can initiate
landing. SDK global sockets/receiver lifecycle remains process-owned; reusing
many live SDK sessions in one process is not a validated use case.

DJITelloPy's cached image and telemetry expose local read time, not verified
sensor capture time, and may repeat stale data. These methods explicitly label
that limitation, and raw telemetry is not marked reliable. A real acquisition
pipeline must establish sensor freshness before anyone uses it for autonomous
motion. Live SDK methods were checked against the official
[DJITelloPy API reference](https://djitellopy.readthedocs.io/en/latest/tello/).
The adapter has been tested only through an injected fake SDK, not hardware.

## Limits and remaining integration work

Monocular depth cannot guarantee free space. Incorrect metric scale, missed
thin obstacles, glass, reflections, unobserved side volume, camera blind spots,
image latency and physical braking all remain material risks. The ROI is an
image heuristic, not a swept-volume collision model. Even 100% valid coverage
means finite predictions, not accurate predictions. The lateral state machine
is a conservative research decision rule requiring later physical validation.

A recording always supplies the same next frame regardless of the generated
command. **Replay is not a physics simulator and cannot prove that a hypothetical
avoidance command would prevent collision.** It can compare observed decisions,
latency sensitivity and failure responses along an already recorded trajectory.

Remaining dependencies are actual Depth Anything 3/Metric3D implementations and
checkpoint conventions, calibrated intrinsics and metric scale validation,
frame-aligned recordings/depth/timestamps, reliable live acquisition and speed
telemetry, measured braking parameters, and separately authorized hardware
validation. No flight experiments or poster assets are included.
