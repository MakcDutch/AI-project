# Real model integration verification

Reviewed on 23 October 2026, on shared `main` at `582005b`.

## Execution status

**No successful real pretrained inference was performed for either model.**
The integration code already connects both wrappers to replay, the controller,
and the mock adapter. This review verified that wiring and its guards, but the
active environment lacks required dependencies and the inspected local caches
contain no pretrained checkpoints. Tello metric-accuracy evaluation is also
blocked by missing matching Tello calibration and real source recordings.

| Check | Depth Anything 3 | Metric3D |
| --- | --- | --- |
| Existing implementation | `src/depth_models/da3.py`, `DA3Metric` | `src/depth_models/metric3d.py`, `Metric3D` |
| Existing CLI selector | `--depth-model da3` | `--depth-model metric3d` |
| Pretrained selection | `depth-anything/DA3METRIC-LARGE` | `yvanyin/metric3d`, `metric3d_vit_small`, `pretrain=True` |
| Wrapper import in active environment | Fails: `No module named 'depth_anything_3'` | Succeeds; does not initialize the model |
| Model code/checkpoint in inspected caches | Not found | Not found |
| Constructor executed in this review | No | No |
| Real prediction shape / finite coverage / plausible values | Not measured | Not measured |
| Real model → controller → mock execution | Blocked, not verified | Blocked, not verified |
| Existing interface/transform tests | Pass with explicitly fake neural backend | Pass with explicitly fake neural backend and CPU Torch |

The two last tests are unit/integration tests of the wrappers' transformation
code, **not evidence that pretrained inference works**. Existing synthetic maps,
point CSVs and PNG result figures were not substituted for neural predictions.
No model constructor was invoked, and no model packages or weights were downloaded or installed.
Only upstream documentation/source text was consulted online.

## Scope and preservation

Initial Git status already contained local work from the preceding integration:

```text
 M docs/software/README.md
 M pyproject.toml
 M tello_nav/cli.py
 M tello_nav/drone/adapters.py
 M tello_nav/replay/runner.py
?? docs/software/INTEGRATION_REVIEW.md
?? examples/create_replay_fixture.py
?? tello_nav/replay/model_provider.py
?? tests/test_model_integration.py
```

Those changes were preserved. No fetch/pull was needed for this local-execution
review, and remote freshness was not rechecked. No branch, commit, push, hardware
connection, flight action or poster work was performed.

This turn adds this report only, plus ignored diagnostic/demo artifacts under
`outputs/real_model_audit/`. There were no changes to model preprocessing,
checkpoint selection, shared adapter, controller, runner, CLI or tests. Existing
coverage was sufficient for unchanged integration code; no duplicate adapters
or new tests using additional fake networks were added.

## Environment and checkpoint inspection

Active Python is `/usr/bin/python`, version 3.12.3. Observed package state:

| Dependency | Observed state |
| --- | --- |
| Torch | `2.14.1+cu130`; import succeeds, CUDA unavailable |
| NumPy / OpenCV | `2.5.3` / `5.0.0.93` |
| Pillow / safetensors | `10.2.0` / `0.8.0` |
| SciPy / pandas / matplotlib | `1.18.1` / `3.0.5` / `3.11.1` |
| `depth_anything_3`, `torchvision`, `huggingface_hub`, `einops` | Absent |
| `timm`, `mmcv`, `mmengine` | Absent |
| DJITelloPy | Absent; unnecessary for mock replay |

CUDA unavailability alone does not establish that inference is impossible:
wrappers accept CPU devices. Actual pretrained CPU compatibility, memory use and
inference time are unverified. No minimum viable environment is inferred from
successful Torch import alone.

The inspected paths were:

```text
/home/marko/.cache/huggingface/hub
/home/marko/.cache/huggingface
/home/marko/.cache/torch/hub
<repository>/models
<repository>/checkpoints
<repository>/weights
```

All six locations were absent. Hugging Face/Torch/XDG cache environment overrides
were considered when resolving paths; `torch.hub.get_dir()` independently
confirmed the Torch path above. No `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`,
`*.onnx` or `*.bin` weights were found at these locations. This is a scoped
cache/repository check, **not a claim that no checkpoint could exist anywhere
else on the computer**. A checkpoint stored elsewhere must be supplied with its
path, expected model version and provenance.

DA3's wrapper delegates initialization to Hugging Face `from_pretrained` for its
fixed model ID. Metric3D delegates to Torch Hub, which needs external model code
as well as pretrained weights. The inspected upstream Hub configuration selects
`metric_depth_vit_small_800k.pth` for ViT-Small. A checkpoint filename alone is
insufficient: architecture/configuration and code revision must also match.
[Metric3D Hub configuration](https://github.com/YvanYin/Metric3D/blob/main/hubconf.py).

The project declares only NumPy/OpenCV and optional live DJI support in
`pyproject.toml`; it does not currently supply a validated model dependency lock.
Upstream DA3 declares Torch, torchvision, Hugging Face Hub, einops and other
packages, with `numpy<2`. This environment has NumPy 2.5.3, so adding only the
missing import would not establish a supported stack.
[DA3 package manifest](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/pyproject.toml).

Metric3D's upstream ViT requirements pin Torch 2.0.1, torchvision 0.15.2,
NumPy 1.23.1 and xformers 0.0.21, alongside additional dependencies. That is not
the current environment, and NumPy 1.23.1 conflicts with this project's
`numpy>=1.24`. Its current Hub loader imports mmcv or mmengine configuration
support. A separately validated dependency set is needed; blindly applying old
pins to this Python 3.12 environment is not a verified solution.
[Metric3D requirements](https://github.com/YvanYin/Metric3D/blob/main/requirements_v2.txt).
No installation/downgrade was attempted.

## Verified input/output contract and existing data flow

```text
VideoReader
  BGR decode -> uint8 RGB Frame(index, timestamp, timestamp_source)
    -> TeamModelDepthProvider
       calibration geometry validation + scalar focal computation
       -> existing model.predict(rgb_copy, focal_px)
       -> aligned HxW validation and invalid/saturation masking
    -> ReactiveController.compute(depth, original_observation_timestamp)
    -> ControlCommand
    -> optional MockDroneAdapter watchdog/send
    -> logger and optional OpenCV visualization
```

The CLI already forwards real-model selection, calibration path and device to
`create_model_provider`; it does not replace either wrapper with the mock depth
provider. Model initialization failures raise an error. Per-frame inference
failures become logged errors and STOP decisions. A log being written, or the
CLI returning zero, **does not by itself prove that any inference succeeded**:
review each record's `error`, sector coverage and `mock_execution` fields.

Both existing wrappers accept `predict(rgb, fx)` with uint8 HxWx3 RGB input and
a scalar focal length in pixels. The shared adapter copies RGB, preserves the
teammates' preprocessing, checks output alignment with the original RGB, and
returns float32 HxW. It does not apply a second metric scaling transformation.

- **DA3:** the wrapper writes RGB as a correctly channel-swapped PNG, passes that
  file to `model.inference`, takes `prediction.depth[0]`, applies the existing
  processed-focal/300 conversion and restores the original image size. The
  metric checkpoint must remain DA3METRIC-LARGE; a relative-only DA3 checkpoint
  is not interchangeable. The documented focal conversion is specific to that
  metric series. [DA3 model FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3#-faq).
- **Metric3D:** the wrapper resizes RGB to fit 616x1064, pads and normalizes it,
  infers in canonical camera space, unpads/resizes, and applies scaled focal/1000.
  Its 0–300m clamp is unchanged. The adapter masks >=300m as unknown, so a positive
  infinity clamped to 300 cannot directly appear as free space.
- **Invalid values:** nonfinite and nonpositive output becomes NaN in the shared
  adapter. Controller sector coverage decides whether remaining observations
  are sufficient; missing/inference-failed depth results in STOP. A finite
  prediction is not proof of accuracy, and wrapper interpolation can already
  have lost confidence/invalid-mask information.
- **Depth convention:** the integration contract is camera optical-axis Z in
  meters, not ray range, inverse depth or relative depth. This interpretation is
  supported by inspected upstream unprojection code, which scales inverse-K
  rays with z=1 by the depth value. That is source-level evidence, not a measured
  validation of either checkpoint in this environment.
  [DA3 geometry](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/geometry.py),
  [Metric3D geometry](https://github.com/YvanYin/Metric3D/blob/main/mono/utils/unproj_pcd.py).

The shared focal convention is the mean of fx/fy times the longest-side resize
ratio, matching `src/calibration/intrinsics.py`. It is preserved for compatibility,
not newly validated as the optimal focal convention for every camera. DA3's
internal resize can round dimensions independently; its wrapper's width-based
processed-focal estimate must be checked against the actual installed API.
[DA3 input processing](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/utils/io/input_processor.py).

No fresh real-model output array exists from this review. Consequently there
are no measured real-model min/max/percentiles, finite-pixel rates, inference
latencies, metric errors or controller outcomes to report.

## Camera calibration and source input

The only available calibration is `results/calib_iphone_960.json`, labeled for
an iPhone with size `[720,960]`, focal entries approximately 669.26 and 664.36
pixels, and nonzero distortion. **It was not used as DJI Tello calibration.**
Its mean focal value is approximately 666.81px at its reference resolution;
that number is not a Tello parameter.

The shared calibration class validates K/size/finite distortion, accepts matching
uniform-resize geometry, and rejects a landscape 960x720 frame for this portrait
calibration. That rejection was verified directly. It neither identifies the
physical camera nor undistorts/corrects crops automatically. The user-selected
calibration must match device, optical settings, orientation, crop, resolution
and distortion treatment.

No raw validation JPGs, real recorded videos or cached full `.npy` model depth
maps are present in `data/` or `results/`. The two PNGs are saved result plots,
not calibrated source frames. The generated fixture videos are labeled synthetic.

**Tello metric accuracy evaluation remains blocked.** No provisional camera
parameters were created. A later technical shape/runtime smoke test may use
explicitly labeled provisional focal parameters if the selected model permits
that use, but the resulting values must not be presented as physically
calibrated meters or evidence of safe control. Current metadata does not
independently certify calibration provenance, so any provisional use must be
clearly labeled in the chosen calibration artifact/scenario and external report.

## Verification performed and exact results

1. Inspected Git status/history, model wrappers, adapter, focal helper, CLI,
   runner, providers/readers, mock boundary, tests and calibration files.
2. Probed the active Python dependency state and local cache/checkpoint paths,
   without constructing either model.
3. Imported the actual wrappers: DA3 failed at its missing package; Metric3D
   imported successfully. An import-only check does not load pretrained weights.
4. Confirmed both CLI model selectors reject initialization without explicit
   model-download permission: **exit 2**, no inference log created.
5. Ran the existing suite:

   ```sh
   python -B -m unittest discover -s tests -q
   ```

   Result: **64 tests run, 64 passed, zero failures/errors/skips**, in 1.962s on
   this run. No new tests were added, and no assertions were weakened. Two tests
   exercise existing wrapper transforms with fake neural backends; none execute
   a real pretrained network.
6. Generated three on-disk synthetic video/depth scenarios and replayed them
   with the actual video reader, NPY provider, controller, logger and mock adapter:
   **36 records, 36 accepted mock commands**. Empty corridor: STOP=1/FORWARD=11;
   center obstacle: BRAKE=10/STOP=2; invalid depth: STOP=12. This is independently
   verified software execution, explicitly **not real-model inference**.
7. `git diff --check` passed. Existing local changes remained present and intact.

Ignored evidence files created by the checks:

```text
outputs/real_model_audit/environment.json
outputs/real_model_audit/imports.json
outputs/real_model_audit/download_gates.json
outputs/real_model_audit/synthetic_summary.json
outputs/real_model_audit/synthetic_replay.json
outputs/real_model_audit/fixtures/scenarios.json
```

## Reproducible commands

Run from the repository root. The tests and synthetic demonstration below run
with the current installed dependencies; no model downloads or real SDK are used.

### Tests and independently runnable replay

```sh
python -B -m unittest discover -s tests -v
python -B -m tello_nav --help
python -B -m examples.create_replay_fixture --output outputs/real_model_audit/fixtures
python -B -m tello_nav --manifest outputs/real_model_audit/fixtures/scenarios.json --mock-drone --output outputs/real_model_audit/synthetic_replay.json
```

The replay command produces `synthetic_replay.json`. Inspect it:

```sh
python -m json.tool outputs/real_model_audit/synthetic_replay.json
```

On a machine with OpenCV GUI support, this existing CLI command displays RGB,
depth, sector statistics, state, command and timestamps. It is a synthetic-data
visualization, not a real-model demonstration. Display opening was not attempted
in the current headless review; rendering is covered by existing tests.

```sh
python -B -m tello_nav --manifest outputs/real_model_audit/fixtures/scenarios.json --mock-drone --visualize --output outputs/real_model_audit/synthetic_visualized.json
```

### Conditional real-model replay commands

These are supported by the implemented CLI but **have not succeeded here**.
They require the missing compatible model environment, pretrained resources,
a real recording and its matching camera calibration. Set the two shell
variables to those actual local paths. The guards below intentionally fail if
paths have not been supplied; no nonexistent example recording/calibration is
invented.

**The existing constructors can fetch external model code and weights.** The
`--allow-model-download` flag in the following commands explicitly authorizes
that behavior. It is not a cached-only/offline guarantee. Do not run these
commands until that behavior is approved. None was executed during this task.
There is currently no CLI `--checkpoint` or guaranteed-offline model-loading
option; such flags must not be assumed to exist.

```sh
: "${REPLAY_VIDEO:?Set REPLAY_VIDEO to the actual recording path}"
: "${CAMERA_CALIBRATION:?Set CAMERA_CALIBRATION to matching camera intrinsics JSON}"

# Depth Anything 3 metric checkpoint; no mock depth backend is selected.
python -B -m tello_nav --video "$REPLAY_VIDEO" --depth-model da3 --calibration "$CAMERA_CALIBRATION" --device cpu --allow-model-download --mock-drone --scenario-id da3-real-inference --output outputs/real_models/da3.json

# Metric3D ViT-Small pretrained checkpoint.
python -B -m tello_nav --video "$REPLAY_VIDEO" --depth-model metric3d --calibration "$CAMERA_CALIBRATION" --device cpu --allow-model-download --mock-drone --scenario-id metric3d-real-inference --output outputs/real_models/metric3d.json
```

Add `--timestamps "$FRAME_TIMESTAMPS"` only if that variable points to the
synchronized `frame_index,timestamp_s` CSV for the same video. Otherwise the
implemented reader labels FPS-derived timestamps as approximate. Offline mode
is the default, so long measured CPU inference does not age observations in the
virtual recording timeline. It cannot establish realtime feasibility. Use
`--mode realtime` for actual timing sensitivity; stale results and the mock
watchdog may correctly cause rejection/STOP.

After a successful model run, inspect both requested commands and mock results:

```sh
python -m json.tool outputs/real_models/da3.json
python -m json.tool outputs/real_models/metric3d.json
```

For real-model RGB/depth/controller display on a GUI-capable machine, repeat one
of the conditional model commands with `--visualize` and a different output path.
For example, after satisfying the same prerequisites and download approval:

```sh
python -B -m tello_nav --video "$REPLAY_VIDEO" --depth-model da3 --calibration "$CAMERA_CALIBRATION" --device cpu --allow-model-download --mock-drone --visualize --scenario-id da3-real-inference-view --output outputs/real_models/da3_visualized.json
```

### Acceptance criteria for the first real run

A future real-model verification must record checkpoint/code/dependency versions,
actual source image identity and calibration provenance. Before claiming success,
inspect the shared adapter's returned array for exact input-aligned HxW shape,
valid finite-positive fraction and depth distribution; inspect all replay errors,
sector coverage, controller decisions and mock acceptance/rejection. A numerical
range alone cannot establish plausibility without the scene, camera and scale.
Verify known distances separately before claiming metric accuracy. Preserve real
predictions and input association if comparing models.

## Remaining blockers and limitations

- Install/validate compatible dependency sets without silently changing the
  shared environment; record exact package and upstream code versions.
- Supply the actual pretrained checkpoints and matching code, or explicitly
  authorize fetching them. Checked locations currently contain neither.
- Supply a real source recording/image and corresponding calibrated geometry.
- Obtain valid Tello calibration before Tello metric accuracy evaluation.
- Execute each real network and verify its outputs and complete replay path;
  none of those real-inference observations is available from this review.
- Consider an explicit cached/local-only loading route after checkpoint
  provisioning; the current CLI gate protects initialization but cannot promise
  no network requests when the flag is enabled.
- Model confidence/sky/invalid-mask propagation and interpolation semantics still
  need coordination; adapter shape validation is not an accuracy certificate.
- Offline replay never changes future recorded frames in response to a command,
  and it cannot prove collision avoidance or flight safety. No physical drone
  operation or autonomous flight capability was validated or executed.
