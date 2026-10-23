# Integration review — 23 October 2026

## 1. Repository Overview

This review inspected the complete tracked source tree, configuration, proposal,
software guide, tests, CSV/JSON artifacts and both result images. Repository
claims below are based on files and commands actually inspected. Model behavior
beyond the exercised wrapper transformations is identified as unverified.

The project has two complementary paths:

```text
Research validation:
calibration photos -> calibration JSON
validation photos -> marker annotations -> geometric Z ground truth
                 -> model wrappers -> point metrics + figures + optional depth cache

Navigation software:
video/synthetic frames -> Frame(index, time, RGB)
                       -> metric DepthProvider
                       -> sector analysis + reactive state machine
                       -> ControlCommand -> JSON/CSV + visualization
                                         -> optional mock adapter execution
```

The proposal in `docs/proposal/main.tex` assigns calibration/ground truth to
Maksym, model evaluation to Yarema, and replay/controller/drone integration to
Marko-Zenon. Git attribution shows Yarema's commit currently supplies calibration
and annotation tools as well as model code. No assumption is made about who
performed the physical measurements. The root README contains only one short
line; `docs/software/README.md` is the operational guide. `data/info.md` is empty.

Detailed inputs, outputs and dependencies:

| Component / files | Responsibility and inputs | Outputs | Dependencies and interaction |
| --- | --- | --- | --- |
| `docs/proposal/main.tex`, `references.bib` | Research scope, methods and responsibilities | Proposal source | LaTeX class/toolchain not part of this software verification |
| `src/calibration/calibrate.py` | Chessboard photos, 7x7 pattern, configured 52mm squares, longest side 960 | K, distortion, RMS, used image names in JSON | OpenCV/NumPy; configured photo directory is absent; script executes at import/top level |
| `src/calibration/check_one.py`, `check_tilt.py` | Corner/tilt diagnostics for configured photos | Printed diagnostics, optional corner image | OpenCV/NumPy; scripts have top-level work and were read, not executed |
| `src/calibration/intrinsics.py` | `load_focal(json_path,img_w,img_h)` | Mean fx/fy times longest-side resize ratio | JSON; consumed by teammate validation; has no aspect/orientation validation itself |
| `src/validation/annotate_markers.py` | `frames.csv` and source photos, eight clicks per image | `markers.csv` with image, marker ID, xy, dimensions | OpenCV GUI; supports skip/undo/resume; underlying photos absent |
| `src/validation/ground_truth.py` | Marker coordinates, calibration, ruler lengths, optional tape distances | `gt_points.csv` containing camera-Z, tape depth, residual, scale source | SciPy/OpenCV/NumPy; joint planar geometry and poses, robust least squares, ruler median scale; distortion disabled by default |
| `src/depth_models/da3.py` | `DA3Metric(device=None)`, `predict(rgb,fx)` | Original-resolution intended metric depth array | Torch, DA3 package, pretrained DA3METRIC-LARGE; adapter consumes existing public method |
| `src/depth_models/metric3d.py` | `Metric3D(variant,device=None)`, `predict(rgb,fx)` | Original-resolution intended metric depth, clamped to 0–300m | Torch Hub and external Metric3D dependencies; adapter consumes existing public method |
| `src/eval/validate.py` | Photos, GT points, calibration, model wrappers or cached maps | Point CSV, summary CSV, error plot, model-specific named `.npy` cache | pandas/NumPy/matplotlib/OpenCV plus model stacks; both models are constructed before cache use |
| `src/eval/show_depth.py` | Source photo, GT points and existing depth caches | RGB/depth comparison PNG | pandas/NumPy/matplotlib/OpenCV; cannot reconstruct maps from point CSV |
| `tello_nav/controller/commands.py` | Four RC channels, state and reason | Immutable validated `ControlCommand` | Standard library; hardware independent |
| `tello_nav/controller/sectors.py` | Numeric HxW metric depth, normalized ROI/boundaries and coverage settings | Three `SectorStats`: percentile, minimum, coverage/counts, known flag | NumPy; invalid values excluded, unknown coverage fails closed |
| `tello_nav/controller/reactive.py` | Depth, original observation timestamp, optional SI telemetry, config/clock | Five-state command and diagnostics | NumPy, sectors/commands; reused unchanged between replay and future live applications |
| `tello_nav/replay/video.py` | OpenCV video, optional synchronized timestamp CSV | Indexed uint8 RGB `Frame` sequence | OpenCV/NumPy; exact external or labeled FPS-approximate timing |
| `tello_nav/replay/providers.py` | Frame through mock, callable, or exact-index `.npy` source | Aligned camera-Z meters | NumPy; protocol and shape validation; mock fixtures are not learned predictions |
| `tello_nav/replay/model_provider.py` (new) | Existing team model, explicit size/K/dist calibration, Frame | Validated aligned depth; invalid/saturated values marked NaN | Lightweight imports until factory explicitly constructs model; does not implement either network |
| `tello_nav/replay/runner.py` | Frame sequence, provider, controller/replay config | Lazy or collected decision records; optional mock execution | Clock/sleep injection; deterministic virtual scheduler, realtime pacing and explicit drop policy |
| `tello_nav/replay/logging.py` | Records and metadata | JSON or CSV plus metadata sidecar | Standard library; JSON rejects nonfinite serialized numbers |
| `tello_nav/replay/visualization.py` | RGB, depth, sectors and decision record | OpenCV display, exit signal | OpenCV GUI; headless rendering tested by patching display calls |
| `tello_nav/drone/adapters.py` | Command plus original observation and command timestamps | Mock command history or explicitly authorized SDK RC requests | Mock has no sockets/SDK initialization; live boundary optionally imports DJITelloPy |
| `tello_nav/cli.py`, `__main__.py` | CLI flags, JSON configs/manifests and file paths | Scenario runs, fingerprints and logs | Installed `tello-replay` entry point or `python -m tello_nav`; no live-drone selection |
| `examples/controller.json` | Explicit controller thresholds/ROI | Example dataclass configuration | Read by CLI |
| `examples/create_replay_fixture.py` (new) | Output directory | Three short synthetic videos, timestamps, aligned mock maps and manifest | OpenCV/NumPy; no model or hardware |
| `tests/test_controller.py`, `test_replay.py`, `test_drone.py`, `test_model_integration.py` | Arrays, generated videos, fake SDKs and fake neural backends | Automated assertions | unittest, NumPy/OpenCV; two new wrapper-body tests optionally use CPU Torch |
| `pyproject.toml`, `.gitignore` | Packaging, core/live dependencies and generated-file exclusions | Installable package, CLI | NumPy/OpenCV required; DJITelloPy optional; research model environments remain separate requirements |

## 2. Teammate Changes

### Git synchronization

- Initial branch: `main`, clean working tree, initially at
  `d84d2e690cc73bac00e01c56ad9a4c3b66eea928`.
- Remote: `origin`, `https://github.com/MakcDutch/AI-project.git`.
- `git fetch origin` found `0` local-only and `2` remote-only commits against
  `origin/main`; there was no divergence or local work to merge.
- `git pull --ff-only origin main` succeeded and advanced the same branch to
  `582005bcfe61581f4ac2f165a0cbbcaecd17c3be`.
- `origin/Yarema` was also discovered, at `1d67c1a`; it was inspected as history,
  not checked out. No feature branch, commit or push was created.
- Fetch required the environment's Git write/network permission; it completed.
  No destructive reset, force update, stash, or conflict resolution was used.

Relevant history:

| Commit | Git author | Meaning |
| --- | --- | --- |
| `d84d2e6` | Marko-Zenon | Previous controller/replay/Tello implementation and original 51 tests |
| `1d67c1a` | yaremkoUCU | Calibration, model wrappers, validation pipeline and first results |
| `582005b` | yaremkoUCU | Merge of Yarema work into shared main |

The teammate commit added **22 files, 1,021 source/text lines**, with no edits
to the prior `tello_nav` implementation or `pyproject.toml`:

- Calibration: four scripts in `src/calibration/` and
  `results/calib_iphone_960.json`.
- Model wrappers: `src/depth_models/da3.py`, `metric3d.py`.
- Annotation/GT: `src/validation/annotate_markers.py`, `ground_truth.py`,
  `data/validation/frames.csv`, `markers.csv`, `gt_points.csv`.
- Evaluation: `src/eval/validate.py`, `show_depth.py`, point and summary CSVs,
  `results/depth_wall_150.png`, `error_vs_distance.png`.
- Empty `data/info.md` and three tracked CPython bytecode files.

The CSVs contain five image references, 40 annotated marker points, 40 GT rows,
80 prediction samples (40 per model), and 16 summary rows. Source `.jpg` images,
calibration photos, videos, model weights, and `results/depth/*.npy` are not
tracked/provided in this checkout. Home-directory model caches were not audited;
no constructor was invoked to discover or fetch weights.

The result images were visually inspected: they show a marked wall scene,
model depth comparisons and distance/error plots. They are saved research
artifacts, not reproducible inference evidence for this machine. The CSV reports
DA3 geometry AbsRel ~0.0959 and Metric3D ~0.1724, including a much larger nearest-wall
Metric3D overestimate. These are **committed results, not newly measured results**;
source photos and full depth arrays are needed to verify them independently.

## 3. Current Implementation Status

| Component | Files | Status | Tested | Remaining Work |
| --- | --- | --- | --- | --- |
| Calibration generation | `src/calibration/*.py` | Implemented research scripts; camera data partial | Source/JSON audit; focal helper cross-check | Restore photos, confirm camera/settings and distortion treatment |
| Marker annotation and GT | `src/validation/*.py`, `data/validation/*.csv` | Implemented; checked-in annotations/results | Source and schema/count inspection | Supply source images; independently validate physical measurements/GT |
| DA3 network wrapper | `src/depth_models/da3.py` | Implemented; connected through new adapter; real inference blocked here | Actual predict body with fake DA3 backend | DA3 package, compatible Torch stack, checkpoint and matching RGB/calibration |
| Metric3D network wrapper | `src/depth_models/metric3d.py` | Implemented; connected through new adapter; real inference blocked here | Actual predict body with CPU Torch and fake network | torchvision/mmcv or mmengine/timm/etc., code/weights, real data |
| Model-to-provider bridge | `tello_nav/replay/model_provider.py` | Implemented and integrated | RGB/focal/shape/mask/factory/calibration tests | Real inference/metric-scale verification and confidence propagation |
| Metric evaluation | `src/eval/*.py`, `results/*.csv/png` | Implemented scripts and saved artifacts; not rerun | Read and visually inspected; no fresh metric claims | Photos/cache/model environment; cache-only behavior; provenance |
| Reactive controller | `tello_nav/controller/*.py` | Implemented, original public interfaces preserved | Original 22 controller/sector tests plus new pipeline checks | Sensor accuracy, swept-volume geometry, measured physical braking |
| Replay and logging | `tello_nav/replay/*.py`, `cli.py` | Implemented; model-selection and mock-execution paths integrated | Original replay tests; new adapter/manifest tests; generated multi-video demo | Replay against actual recorded scenes and full model predictions |
| Synthetic fixtures | `providers.py`, `examples/create_replay_fixture.py` | Implemented mocks, explicitly not learned | Eight existing scenarios, three generated-video scenarios | Actual adverse-surface recordings |
| Mock drone | `drone/adapters.py`, replay runner | Implemented and integrated end to end | Fake SDK/adapter tests, accepted/rejected replay commands | No physical validity claim |
| Tello live boundary | `drone/adapters.py` | Implemented, authorization gated; hardware unverified | Fake SDK only; no physical commands | SDK install, sensor-time acquisition, independent operator procedure |
| Flight experiments/poster | No implementation added | Deferred | Not run | Separate future task and flight authorization |

## 4. Depth Model Integration

### Existing public interfaces and conventions

Both wrappers take **uint8 RGB HxWx3** and a scalar focal length in image pixels,
then return a nominal metric HxW NumPy array. The `fx` name is misleading in the
current research pipeline: `load_focal` actually averages K's fx and fy. The new
adapter preserves that convention, rather than silently switching the experiment
to a different focal value or adding an empirical correction.

DA3 uses `DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")`,
selects CUDA when available unless a device is supplied, writes a temporary PNG,
calls `model.inference([path])`, takes `prediction.depth[0]`, scales the input focal
by processed-output width/original width, applies `/300`, and bilinearly restores
the original image dimensions. Its wrapper does not expose intrinsics, confidence,
sky masks or preprocessing options to the caller. The upstream API uses resize
and normalization internally; default resize behavior must remain compatible
with the wrapper's width-ratio assumption. No explicit `.eval()` appears in the
wrapper, although the inspected upstream API initializes its internal model in
evaluation mode. Actual installed DA3 behavior was not tested because DA3 is absent.

Metric3D uses `torch.hub.load("yvanyin/metric3d", "metric3d_vit_small",
pretrain=True, trust_repo=True)`, puts the model in evaluation mode, and globally
disables xformers module imports in its constructor. It resizes to fit 616x1064,
pads with RGB normalization means, normalizes channels using its mean/std,
executes `model.inference({"input": x})` under `no_grad`, removes padding,
bilinearly restores HxW, multiplies by resized focal/1000, and clamps to 0–300.
There is small integer-resize/focal-rounding uncertainty in its nominal resize
factor; its public behavior was not changed. Confidence returned by inference is
discarded. CUDA is optional in the wrapper, but upstream CPU performance and
compatibility with this machine's Torch version have not been verified.

The metric focal factors agree with the official
[DA3 metric-model FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3#-faq)
and [Metric3D inference example](https://github.com/YvanYin/Metric3D/blob/main/hubconf.py).
The adapter **does not apply those conversions again**. Neither leave-one-frame-out
global scaling from the evaluator nor pointwise CSV values are fed into control.

The **optical-axis Z interpretation** is supported by inspection of upstream
unprojection code: DA3 multiplies inverse-K pixel rays (z=1) by depth, and
Metric3D reconstructs points with a z=1 base ray. This is an inference from
[DA3 geometry source](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/geometry.py)
and [Metric3D unprojection source](https://github.com/YvanYin/Metric3D/blob/main/mono/utils/unproj_pcd.py),
not a measured calibration result. The team's GT code also projects and writes
camera-space `X[:,2]`. Relative-only checkpoints or range outputs cannot be
substituted for these wrappers without a separately justified adapter.

### New integration boundary

`TeamModelDepthProvider` wraps an existing model object and validated
`CameraCalibration`, invokes its original method, checks aligned HxW output,
and normalizes invalid values to NaN. Inputs are copied to protect recordings
from an implementation that mutates its input. No model implementation changed.

Metric3D's clamp converts positive infinity to finite 300m. The new adapter marks
**all values >=300m unknown**, including legitimately distant saturated estimates.
This intentionally sacrifices far-range coverage to avoid treating that numerical
failure as clearance. Neither adapter can recover invalid/confidence information
already lost during a wrapper's interpolation or clamping; that remains a model
API improvement for coordination with Yarema.

`CameraCalibration` validates size, K and finite distortion coefficients,
requires matching aspect/orientation up to resize rounding, and preserves the
research focal helper convention. It does not undistort images, transform crops,
infer camera identity, or certify that an explicitly supplied JSON belongs to
the current camera. The checked-in file is **iPhone-labeled**, portrait 720x960,
not a Tello calibration. Its RMS is a saved fit statistic, not a safety guarantee.
The new boundary rejects applying that portrait geometry to the landscape demo.

CLI selectors are `--depth-model da3` and `--depth-model metric3d`, with explicit
`--calibration`. `--device` defaults to CPU. The existing constructors can fetch
large weights and remote code, so `create_model_provider` refuses to initialize
unless `allow_model_download=True` (CLI `--allow-model-download`). That permission
was **not used for real models during this review**. Applications with an already
initialized local model may directly construct `TeamModelDepthProvider` instead.

A manifest may mix providers across scenarios; each model entry carries its own
calibration path. Identical model/calibration/device combinations reuse the
provider; each scenario still gets a new controller and mock adapter.

### Environment and blockers

Observed locally: Torch `2.14.1+cu130`, CUDA unavailable. NumPy/OpenCV, pandas,
SciPy and matplotlib are importable. `torchvision`, `depth_anything_3`, `mmcv`,
`mmengine`, `timm` and `djitellopy` are absent. The Metric3D wrapper imports, but
its constructor was intentionally not run. DA3 wrapper import fails specifically
with `No module named 'depth_anything_3'`.

Teammate changes did not declare model/evaluation dependencies in `pyproject.toml`.
Do not blindly install old upstream pinned requirements into this environment:
[Metric3D's requirements](https://github.com/YvanYin/Metric3D/blob/main/requirements_v2.txt)
include a different Torch/NumPy stack from this project's current environment.
A mutually compatible, pinned environment/checkpoint/code revision needs team
validation. Core replay tests remain independent of these model stacks.

The package finder now includes the existing `src`, `src.calibration`, and
`src.depth_models` namespace packages so installed adapters can resolve the
teammates' modules; source paths/public model APIs were not reorganized.

## 5. Reactive Controller Review

The controller crops the normalized ROI and divides its width at configured
left/center/right boundaries. Sector coverage uses finite positive samples;
unknown sectors STOP even if other directions look clear. The 10th percentile
(default) reports robust proximity, but the minimum sample also vetoes clearance,
so a thin valid obstacle below percentile support cannot disappear in averaging.
The tradeoff is susceptibility to false stops from near-valued noise.

Initial/recovery STOP needs consecutive center-clear frames above the release
threshold. FORWARD uses the base threshold; close center depth enters zero-RC
BRAKE. BRAKE enforces a dwell before choosing an opt-in clear side. Avoidance
holds direction while safe, immediately brakes if its chosen side becomes unsafe,
and brakes again before forwarding after center recovery. There is no direct
left/right direction reversal. Missing, malformed, unknown, non-increasing,
future or stale observations STOP and reset recovery.

Commands stay within configured integer SDK bounds. Dynamic clearance adds
speed x (observation age + reaction latency) and braking distance only for fresh,
explicitly reliable SI telemetry and positive configured deceleration; arithmetic
overflow stops. Raw Tello fields do not satisfy that reliability contract.

No new controller defect requiring a behavioral change was demonstrated by this
review; its implementation is unchanged. Remaining assumptions matter:

- Finite model predictions are not necessarily correct. A thin obstacle predicted
  as distant remains invisible to these checks.
- Coverage below 100% intentionally permits some holes; invalid regions outside
  the crop are unobserved. Default 90% is a research setting, not proven safe.
- Pixel sectors do not represent a swept aircraft volume; body/side clearance is
  unobserved. Lateral motion remains disabled by default.
- BRAKE/STOP request zero RC; dwell does not measure actual stopping.
- Base thresholds and reliable braking telemetry require physical calibration.

## 6. Replay Stand Review

The video reader yields original indices and RGB, labels FPS-derived timestamps
as approximate, supports strict external CSV timestamps, and validates decode/count
mismatches where metadata permits. NPY loading uses exact indices and fails
closed on missing files or wrong shapes. Team validation caches use different
`model__image_stem.npy` names; there is no safe automatic video-index mapping
from the sparse point CSV, so no such mapping was invented.

Offline decisions use recording-relative time and ignore measured runtime.
Virtual mode adds configured delays through a deterministic sequential queue.
Realtime mode paces original timestamps and includes actual processing plus
simulated delays in observation age. Late frames are either processed with
freshness checks or explicitly dropped with STOP. Communication delay is applied
before the final safety decision, not retroactively to a fresh command.

`iter_records` is lazy and supports stepping; `--step` is not realtime-compatible.
Manifests reset per-scenario controller state. Logs include sectors/coverage,
commands/reasons, measured and simulated latency separately, frame age, safety
and drop/error events; CLI records config and input SHA-256 fingerprints.
Measured runtime is observational and excluded from decision determinism.

The new `--mock-drone` path calls the mock watchdog and command validator in the
same replay-relative clock domain. It records `accepted`, `inhibited`, error and
`effective_command` separately from the controller's requested command. The
mock adapter keeps its independent default .3s freshness bound; the controller's
.5s threshold does not override it. A stricter adapter may reject a controller
command. Timeout latches inhibition, so a long pause/inference needs an operator
reset or a new scenario. Mock watchdog checks occur at decision boundaries;
realtime between-frame watchdog behavior is covered separately by fake-SDK
background-thread tests, not modeled continuously by offline replay.

**Fixed video frames never change in response to a command.** The stand compares
perception/decisions along a recorded trajectory; it cannot establish the actual
alternative trajectory, collision avoidance success, or flight stability.

## 7. End-to-End Integration

Verified paths:

```text
synthetic RGB -> mock metric depth -> actual controller
              -> JSON/CSV + actual MockDroneAdapter

generated on-disk video + external timestamps + aligned synthetic NPY
              -> actual VideoReader -> NpyDepthProvider -> controller
              -> JSON + actual MockDroneAdapter

RGB -> teammate predict() body with fake neural backend
    -> calibration/model adapter -> controller -> logger/mock execution
```

The last path exercises actual wrapper preprocessing/postprocessing with known
fake output, not a learned network. The interface is connected for both DA3 and
Metric3D, but the full **real RGB -> real model -> control** path remains blocked
by the environment/data/calibration dependencies listed above.

The live adapter remains separately gated by live mode and operator authorization.
It validates bounds and two timestamps, has a background watchdog and latched
manual/emergency STOP, and catches connection/send/video/telemetry failures.
This review added rejection of older command/observation timestamps inside an
otherwise fresh window, preventing delayed results replacing newer decisions.
No real SDK instance was constructed. There is no takeoff/landing/autonomous
flight runner in this task. Cached live video/telemetry read time still cannot
certify sensor capture freshness, and zero UDP RC is not guaranteed physical stop.

## 8. Test Results

Baseline immediately after fast-forward:

```sh
python -m unittest discover -s tests -q
```

Result: **51 tests passed**. Existing assertions were preserved.

Expanded suite:

```sh
python -m unittest discover -s tests -v
```

Result: **64 tests passed**, zero failures/errors/skips in this environment.
There are 13 new tests: 11 adapter/integration tests and two existing-wrapper
transformation tests. The two latter tests use CPU Torch and are explicitly
skipped if optional Torch is unavailable; the other 62 do not require Torch,
GPU, network, weights or physical hardware.

The added checks cover calibration validity/aspect/focal parity, RGB copying,
shape/alignment, no double scaling, invalid/saturated values, model-download
gates, actual constructor signatures through fakes, dependency errors, model
provider-to-controller-to-mock flow, failures/delays, fresh-but-out-of-order
commands, multi-model manifests and original timestamp preservation. Existing
tests cover hysteresis, braking, thin obstacles, all synthetic scenarios,
video alignment, replay timing/determinism, GUI rendering and fake-SDK failures.

An initial expanded run exposed a **test harness issue**: restoring a patched
`sys.modules` snapshot unloaded newly imported Torch, and its second native
extension initialization failed/crashed. The fixture now imports Torch before
that snapshot and preserves its native module lifetime. Model/controller code
and assertions were not weakened to conceal the failure; the corrected full
suite passes. Tests also disable bytecode writes while loading teammate modules,
to avoid dirtying the three already tracked `.pyc` files. Any bytecode rewritten
during early inspection was restored to its exact original HEAD bytes.

No real-inference accuracy/performance tests ran. Saved metrics were not
regenerated. GUI rendering was tested without a physical display. Hardware tests
used fake SDK objects only.

The local no-dependency wheel build also passed, and its archive contains both
team wrappers, `src/calibration/intrinsics.py` and the new model provider. Syntax
parsing and `git diff --check` passed. A final diff check confirmed no changes to
teammate source, data/results, the original three test files or proposal/root README.

The generated multi-scenario demo produced **36 records, all 36 accepted by the
mock boundary**: empty corridor STOP=1/FORWARD=11, center obstacle BRAKE=10/STOP=2,
invalid depth STOP=12. Repeated BRAKE/STOP for a persistently blocked scene is
zero-motion reconsideration, not physical obstacle avoidance. A virtual delay
run with .35s inference + .2s communication produced **30 STOP decisions** and
mock inhibition on every frame. These are fixture outcomes, not drone metrics.

## 9. Problems Found and Changes Made

| Problem | Resolution / files | Verification |
| --- | --- | --- |
| Model API `predict(rgb,fx)` did not match `predict(Frame)` | Added `tello_nav/replay/model_provider.py`; unchanged teammate wrappers are reused | RGB/focal/shape and actual wrapper-body tests |
| No calibrated CLI model selection or explicit weight-fetch gate | Updated `tello_nav/cli.py`: depth model, calibration, device, download permission; manifest support and provider reuse | Constructor/gate/error and multi-model manifest tests |
| Calibration helper accepted arbitrary orientation/aspect | New adapter validates geometry before per-frame prediction, preserving original mean-focal convention | Helper parity and mismatch tests |
| Metric3D +inf could become finite 300m clearance | New adapter masks saturated >=300m as unknown | Actual wrapper clamping test and invalid-depth STOP checks |
| Replay only logged commands rather than executing through mock boundary | Updated `runner.py` with optional mock execution/diagnostics and provider metadata | Deterministic acceptance/failure/delay tests and 36-record demo |
| Old but still fresh commands could replace newer ones at adapter boundary | Updated `drone/adapters.py` with timestamp-order checks and explicit reset of ordering state | Out-of-order regression test |
| Installation only discovered `tello_nav`, not required teammate modules | Updated `pyproject.toml` package discovery, no new required dependencies | Local wheel contents checked |
| No supplied videos/depth for reproducible recorded-input demonstration | Added `examples/create_replay_fixture.py`, clearly synthetic | Generated videos/NPY/timestamps consumed through real reader and mock adapter |
| Software guide described models as absent | Updated `docs/software/README.md`, added this report | Commands and implementation cross-checked |
| Tracked bytecode and incomplete model provenance/data docs | Reported; source/bytecode/data left unchanged | Final diff audit |

Modified existing files: `pyproject.toml`, `tello_nav/cli.py`,
`tello_nav/replay/runner.py`, `tello_nav/drone/adapters.py`,
`docs/software/README.md`.

New files: `tello_nav/replay/model_provider.py`,
`tests/test_model_integration.py`, `examples/create_replay_fixture.py`,
`docs/software/INTEGRATION_REVIEW.md`.

Generated fixtures/logs are under ignored `outputs/integration_demo/`.
No teammate `src` implementation, annotation/calibration/result artifact,
existing test, proposal or root README was changed.

## 10. Remaining Tasks

### Critical

1. Supply actual RGB recordings/photos and full frame-aligned depth maps/timestamps.
   Point samples and result figures cannot substitute for depth maps.
2. Establish a compatible model environment; approve any external code/weight
   fetch separately, record code/checkpoint revisions, and run real inference.
3. Supply matching Tello intrinsics, geometry and acquisition provenance. The
   iPhone JSON must not silently become Tello calibration.
4. Verify metric scale and Z conventions empirically on known distances before
   using model predictions for any physical decisions.

### Important

1. Coordinate confidence/sky/invalid-mask propagation and interpolation policy
   with Yarema; full valid coverage is not confidence or correctness.
2. Validate distortion, focal averaging, integer resize effects, crop/orientation
   handling and the observed near-distance Metric3D overestimate on real inputs.
3. Pin model/evaluation dependencies and checkpoints, add dataset/recording
   manifests and source provenance. Existing point CSV/cache filenames need an
   explicit mapping before any video reuse.
4. Evaluate cache-only operation without model construction and avoid loading
   both networks unnecessarily in `src/eval/validate.py`. This teammate script
   was not rewritten as part of the integration boundary.
5. Remove tracked bytecode through a coordinated repository-cleanup change;
   `.gitignore` already ignores future bytecode but does not untrack these files.
6. Before later live autonomy, establish actual sensor capture timestamps,
   trustworthy telemetry, measured braking and independent operator control.

### Optional

1. Add a reader for the team's annotated image list once source images exist.
2. Extend replay diagnostics/streaming logs for long datasets and interruptions.
3. Make research script paths/configuration reusable instead of hardcoded;
   add focused GT/evaluator unit tests coordinated with their owner.
4. Support explicit frame-to-cache manifests and richer model confidence APIs.

### Deferred flight experiments and poster

No flight experiments, autonomous sorties, collision tests, physical movement,
poster work or automatic commits/pushes were performed. Those remain future
work outside this software review.

## 11. Reproduction Instructions

All commands below run from this repository root. They use implemented commands
and paths that exist or are generated by the preceding command.

```sh
python -m unittest discover -s tests -v
python -m tello_nav --help
python -m tello_nav --synthetic empty_corridor --mock-drone --output outputs/integration_demo/synthetic.json
python -m examples.create_replay_fixture
python -m tello_nav --manifest outputs/integration_demo/scenarios.json --mock-drone --output outputs/integration_demo/replay.json
python -m tello_nav --video outputs/integration_demo/empty_corridor/video.avi --timestamps outputs/integration_demo/empty_corridor/timestamps.csv --depth-dir outputs/integration_demo/empty_corridor --mock-drone --output outputs/integration_demo/single.csv
python -m tello_nav --synthetic empty_corridor --mock-drone --mode virtual --inference-delay .35 --communication-delay .2 --output outputs/integration_demo/delayed.json
```

For display, add `--visualize` to a replay command on a GUI-capable machine.
For terminal stepping, add `--step` in offline/virtual mode. These controls do
not alter subsequent observations.

The following existing-file command intentionally demonstrates the model gate;
it is expected to exit 2 before any model import/download:

```sh
python -m tello_nav --video outputs/integration_demo/empty_corridor/video.avi --depth-model da3 --calibration results/calib_iphone_960.json --output outputs/integration_demo/model_gate.json
```

It is **not** a real model demo: the video is synthetic and the portrait iPhone
calibration does not match it. Do not bypass the gate and treat that combination
as a calibrated test. A genuine inference command must use the team's actual
recording and matching calibration, select one of the implemented depth-model
flags, and explicitly authorize constructor downloads if needed. No successful
real-model invocation can be supplied from the currently available input files.

To check packaging without fetching dependencies or model weights:

```sh
python -m pip wheel --no-build-isolation --no-deps --no-cache-dir --wheel-dir /tmp/tello-nav-review-wheel .
```

Standard build tools must already be installed. Inspection and synchronization
commands used for this review were `git status`, `git branch --show-current`,
`git remote -v`, `git log --oneline -15`, `git fetch origin`,
`git rev-list --left-right --count HEAD...origin/main`, and
`git pull --ff-only origin main`. Do not rerun synchronization over local work
without first rechecking Git status.
