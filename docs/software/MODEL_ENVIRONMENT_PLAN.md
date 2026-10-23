# Model environment assessment and installation plan

Assessment date: 23 October 2026. Repository: shared `main`, HEAD `582005b`.

**This is a plan, not an installation or a successful inference report.** No
packages, interpreters, upstream repositories or checkpoints were installed or
downloaded. The only network reads were official documentation/source and weight
listing pages, not weight payloads. No teammate code, previous local changes,
branches or commits were modified. This report is the only new source-tree file
from this assessment; read-only source hashes are under ignored
`outputs/model_environment_assessment/source_hashes.json`.

## 1. Machine and environment: confirmed observations

| Item | Observed value | Implication |
| --- | --- | --- |
| OS | Ubuntu 24.04.5 LTS, x86-64, Linux 6.17.0-40-generic | Linux CUDA wheels are appropriate |
| CPU | AMD Ryzen 7 7735HS, 8 cores / 16 threads, AVX2 | CPU preprocessing available; no realtime inference rate established |
| Physical RAM visible to OS | 15,590,128 KiB = 14.87 GiB | Commonly sold as a 16 GB configuration; use OS-visible total for planning |
| Available RAM during inspection | Approximately 4–4.5 GiB | Close memory-heavy applications before loading a network |
| Swap | 4 GiB total; about 2.4 GiB already used | Avoid simultaneous models; swapping is not usable GPU memory |
| NVIDIA GPU | GeForce RTX 3050 **6GB Laptop GPU** | Actual discrete CUDA GPU exists |
| Nominal VRAM | 6,144 MiB; first query reported 5,790 MiB free | Capacity/free memory must be rechecked before a run |
| PyTorch-reported device memory | 6,086,262,784 bytes total (5.67 GiB), 5,976,227,840 bytes free at query | Runtime reports differ from nominal NVML total; budget against runtime availability |
| Compute capability | 8.6 | Confirmed by host PyTorch query |
| Other GPU | Integrated AMD Radeon 680M; sysfs reports 512 MiB VRAM and shared-memory aperture | Not the target of the current CUDA wrappers; no ROCm route was validated |
| Loaded NVIDIA driver | 580.173.02, open kernel module | Host `nvidia-smi` communicates successfully |
| `nvidia-smi` CUDA field | 13.0 | Driver capability, not the installed compiler version |
| Installed CUDA compiler/toolkit | nvcc 12.8.93 / CUDA toolkit 12.8.2 | Separate from PyTorch's bundled runtime |
| Current Torch | 2.14.1+cu130, built for CUDA 13.0 | CUDA works on the host outside the restricted sandbox |
| BF16 device support | `torch.cuda.is_bf16_supported()` returned True | Relevant to DA3's upstream autocast path; not an inference result |
| Free disk | About 122 GiB available on the 468 GiB filesystem | Repository, home and `/tmp` share this disk; do not add their free-space values |
| Active Python | `/usr/bin/python`, CPython 3.12.3, no venv active | Do not mutate system/user site-packages for model experiments |
| Environment tooling | `python3.12-venv` installed; uv 0.12.23; git/gcc/g++ available | Existing Python 3.12 can create isolated environments without another interpreter download |

Hardware commands included `lscpu`, `free -h`, `df -h`, `lspci -nnk`, driver/sysfs
reads, `nvcc --version`, `nvidia-smi`, and Torch CUDA property queries. Driver
570 and 580 package records coexist, but the loaded kernel module and working
host query both identify **580.173.02**. Do not infer a broken driver merely from
installed package names; no driver update is recommended by this assessment.

### Important correction to earlier environment reports

Inside the restricted execution sandbox, `nvidia-smi` cannot communicate with
the driver and Torch reports CUDA unavailable. The same read-only queries run
outside that sandbox succeed, including Torch device enumeration, capability,
free-memory and BF16-support queries. Thus **sandbox GPU visibility is restricted;
the laptop is not GPU-less and host CUDA is not broken**. Run future model tests
from a normal host terminal or an explicitly GPU-accessible execution context.
Installing packages cannot fix sandbox device isolation.

The R580 driver is compatible with older CUDA applications; a CUDA 12.4 PyTorch
wheel does not require removing the installed CUDA 12.8 toolkit. Building custom
extensions is a separate matter and may require matching build toolchains.
[NVIDIA CUDA driver compatibility](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

## 2. Current dependencies and repository declarations

Local `pyproject.toml` requires Python >=3.10, NumPy >=1.24, OpenCV >=4.8 and
setuptools >=68 for builds. Its optional `live` extra is DJITelloPy >=2.5; do
not install it for this task. There is no model dependency lock or model extra.
Package discovery includes the existing `src.calibration` and `src.depth_models`
namespaces as well as `tello_nav`.

Installed now:

| Package | Version |
| --- | --- |
| torch | 2.14.1+cu130 at runtime |
| numpy | 2.5.3 |
| opencv-python | 5.0.0.93 |
| Pillow | 10.2.0 |
| safetensors | 0.8.0 |
| scipy / pandas / matplotlib | 1.18.1 / 3.0.5 / 3.11.1 |
| setuptools / pip | 84.2.0 / 24.0 |

Absent: `depth_anything_3`, `torchvision`, `huggingface_hub`, `einops`, `timm`,
`mmcv`, `mmengine`, and `xformers`. A Torch-only environment is insufficient.
The inspected Hugging Face/Torch caches and repository `models`, `weights` and
`checkpoints` paths are absent. No checkpoint is available at those locations;
this does not rule out a user-managed file elsewhere on the machine.

## 3. Existing model implementations and exact requirements

### Depth Anything 3

`src/depth_models/da3.py` imports OpenCV, NumPy, Torch and
`depth_anything_3.api.DepthAnything3`; standard-library imports handle its PNG
scratch file. Its constructor uses exactly:

```python
DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
```

The current wrapper offers only `device`, not checkpoint path, revision,
resolution or architecture selection. Do not substitute a smaller any-view or
relative-only model. The existing CLI `--depth-model da3` selects this wrapper.

Input is uint8 HxWx3 RGB. The wrapper writes a channel-correct PNG, calls DA3
inference, takes `prediction.depth[0]`, applies processed focal x raw depth / 300,
and restores HxW. The upstream metric-model FAQ documents this focal conversion;
it does not apply unchanged to every DA3 variant. The adapter performs no second
scaling. [DA3 model table and FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3#-faq).

Confirmed upstream manifest requirements include `torch>=2`, `torchvision`,
`einops`, `huggingface_hub`, `numpy<2`, OpenCV, Pillow, safetensors, omegaconf,
trimesh, imageio, xformers, open3d, fastapi, uvicorn, requests, typer>=0.9.0,
evo, e3nn, moviepy==1.0.3, plyfile, pillow_heif, pycolmap and pre-commit.
Python is declared >=3.9 and <=3.13. The upstream package version is 0.0.0;
record a source Git SHA as well as its package version. Most dependencies have
**no exact upstream version pin**, so an exact validated full lock cannot be
claimed from the manifest alone.
[Official package manifest](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/pyproject.toml).

The upstream API imports export modules even for normal inference. Installing
only Torch plus the DA3 Python file is not a reliable minimal setup. Start with
the basic package under constraints; do not select the optional Gradio `app`,
Gaussian `gs`, or `all` extras. Its inspected attention implementation uses Torch
scaled-dot-product attention, while the package still declares xformers.
[API imports](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py),
[attention implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/dinov2/layers/attention.py).

### Metric3D

`src/depth_models/metric3d.py` imports OpenCV, NumPy and Torch. Construction uses:

```python
torch.hub.load("yvanyin/metric3d", variant, pretrain=True, trust_repo=True)
```

The wrapper accepts a variant, but the existing adapter/CLI selects exactly
`metric3d_vit_small`. Upstream also exposes ViT-Large, ViT-Giant2, ConvNeXt-Tiny
and ConvNeXt-Large. Its ViT path uses 616x1064 RGB resize/pad/normalization,
canonical inference, unpadding/upsampling, and resized focal/1000 scaling before
a 0–300m clamp. ConvNeXt's documented input size is different; do not change the
variant in this ViT wrapper merely because another name contains “tiny.”
[Official Hub definitions and inference example](https://github.com/YvanYin/Metric3D/blob/main/hubconf.py).

The upstream ViT requirements **do pin** Torch 2.0.1, torchvision 0.15.2,
NumPy 1.23.1 and xformers 0.0.21. They also list Pillow, OpenCV, DateTime,
matplotlib, plyfile, HTML4Vision, timm, tensorboardX, imgaug, iopath,
imagecorruptions and mmcv without exact pins.
[Official requirements](https://github.com/YvanYin/Metric3D/blob/main/requirements_v2.txt).

Those historic pins are not directly appropriate to the existing Python 3.12
project: NumPy 1.23.1 violates the project's >=1.24 requirement, and an old
Torch/Python wheel combination cannot be assumed available. Do not install that
whole file into the current environment or pretend a modern candidate is the
upstream-pinned environment.

Important transitive imports:

- Hub config offers mmcv/mmengine fallback, but `mono/utils/comm.py` directly
  imports `mmcv.utils.collect_env`. **mmengine alone does not satisfy this path.**
  Use a candidate `mmcv==1.7.2` without compiled ops, not arbitrary mmcv 2.x.
  [Communication utilities](https://github.com/YvanYin/Metric3D/blob/main/mono/utils/comm.py).
- Backbone package initialization imports ConvNeXt even when selecting ViT, so
  `timm` is needed; old module paths remain mapped in timm 0.9.16.
  [Backbone imports](https://github.com/YvanYin/Metric3D/blob/main/mono/model/backbones/__init__.py),
  [timm release metadata](https://pypi.org/project/timm/0.9.16/).
- Metric utilities import matplotlib. This dependency is needed even without
  running the project's accuracy evaluation script.
- The teammate wrapper intentionally disables xformers imports globally before
  constructing Metric3D. Upstream has a plain Torch attention fallback; installing
  xformers into this environment would not make this wrapper use it.
- Upstream RAFT decoder allocates bins and grids on CUDA explicitly. The wrapper
  accepting `device="cpu"` **does not establish CPU support**. Use host CUDA for
  the unchanged Metric3D implementation. CPU use would require a separate,
  coordinated code change or validated upstream revision, outside this plan.
  [Decoder CUDA allocations](https://github.com/YvanYin/Metric3D/blob/main/mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py).

## 4. Smallest appropriate checkpoints and official sources

| Initial choice | Required resources | Published checkpoint size | Why this selection |
| --- | --- | --- | --- |
| DA3METRIC-LARGE | HF `depth-anything/DA3METRIC-LARGE`, `config.json` and `model.safetensors` | **1.34 GB** | Existing fixed metric wrapper; no smaller metric-only DA3 checkpoint in the inspected official table |
| Metric3D ViT-Small | Official Metric3D Hub code/configs plus `metric_depth_vit_small_800k.pth` | **150 MB** | Smallest ViT configuration supported by the existing adapter; avoids unnecessary architecture/input-size changes |

The sizes are published decimal sizes, not measured runtime RAM/VRAM and not
precision/quantization targets. Only listing pages were read:
[DA3 weights](https://huggingface.co/depth-anything/DA3METRIC-LARGE/blob/main/model.safetensors),
[Metric3D weights](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_small_800k.pth).

Published SHA-256 values at inspection:

```text
DA3 model.safetensors:
bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776
Metric3D metric_depth_vit_small_800k.pth:
b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6
```

Verify hashes and record model/code revisions when downloading is later approved.
A changed upstream main revision requires rechecking the listed hashes. DA3's
config and Metric3D's architecture/config files are required alongside weights.
The current constructors choose remote IDs/default repository revisions rather
than accepting these exact hashes as arguments.

DA3-SMALL, DA3-BASE and DA3MONO-LARGE must not silently replace DA3METRIC-LARGE:
smaller relative/any-view weights do not preserve the present metric contract.
The much larger nested DA3 model is unnecessary for this initial single-frame
technical test and has different scaling behavior.

## 5. Resource assessment: facts versus estimates

**Confirmed:** the checkpoint payloads total about 1.49 GB; disk has about 122 GiB
free; host CUDA sees the GPU; its nominal capacity is 6 GiB. No model weights
were loaded, so **there is no measured peak inference RAM/VRAM or FPS**.

Planning allowances below are estimates, not official minimum specifications:

| Resource | Conservative planning allowance | Qualification |
| --- | --- | --- |
| Disk for two CUDA environments, dependency/download caches, source and weights | Reserve **25–40 GiB** | Transitive wheels and duplicate caches can dominate the 1.49 GB of weights; check after resolver reports |
| Available host RAM before DA3 load | Aim for **at least 8 GiB free/available** | Loading may hold model plus checkpoint and libraries; not a measured minimum |
| Available host RAM before Metric3D load | Aim for **4–6 GiB available** | Small checkpoint, but framework/preprocessing/activation overhead remains |
| DA3 single-view VRAM | Budget most of the approximately 5.7 GiB runtime-visible GPU memory | Batch 1/default upstream 504 longest-side preprocessing is a plausible smoke-test candidate; fit is unverified |
| Metric3D ViT-Small VRAM | Start with the GPU otherwise idle; retain the full available budget | Disabled memory-efficient attention and 616x1064 input can dominate small weight size; no reliable numerical peak was found |

The GPU is a credible target for sequential smoke tests, **not proof both models
fit or run fast enough**. Start with Metric3D ViT-Small, then a separate DA3 process.
Avoid a manifest holding both models simultaneously. The shared CLI caches model
providers within one run, so merely switching scenarios may retain both networks.

DA3 upstream uses inference mode/autocast and can select BF16 on this GPU. Metric3D's
wrapper uses no-grad but does not enable mixed precision. Do not blindly call
`.half()` or change inference resolution to force a fit: those are separate
behavior/performance changes requiring verification. Small source images do not
reduce Metric3D's fixed internal 616x1064 input. On OOM, capture evidence and
coordinate a controlled change or use a larger machine; do not change metric
models or scaling silently. CPU fallback is uncertain for DA3 and specifically
not established for the unchanged Metric3D decoder.

## 6. Recommended installation strategy

Use **two new isolated environments** and keep the current 64-test environment
untouched. Python 3.12.3 is already available; start there rather than downloading
another interpreter. Two processes/environments separate dependency experiments
and Metric3D's global xformers disabling. They do not require two simultaneously
resident networks.

Recommended **candidate pins**, not a tested full model lock:

| Package | Candidate pin | Evidence / intent |
| --- | --- | --- |
| torch / torchvision | **2.5.1+cu124 / 0.20.1+cu124** | Official paired releases support Python 3.12; established CUDA 12.4 binaries work with host R580 driver |
| numpy | **1.26.4** | Satisfies project >=1.24 and DA3 <2; avoids NumPy-2 compatibility problems |
| opencv-python | **4.11.0.86** | Candidate older wheel compatible with NumPy 1.x; do not let latest OpenCV force NumPy 2 |
| setuptools / wheel | **70.3.0 / 0.45.1** | Candidate build tooling; setuptools retains pkg_resources needed by legacy mmcv setup |
| xformers, DA3 environment only | **0.0.28.post3** | Official changelog pairs its binary wheels with Torch 2.5.1; prebuilt Linux CPython-3.12 files exist |
| mmcv, Metric3D environment only | **1.7.2**, ops disabled | Contains legacy `mmcv.utils`; pure-Python build path avoids custom CUDA compilation |
| timm / matplotlib, Metric3D environment | **0.9.16 / 3.9.2** | Candidate inference-import dependencies, not upstream exact requirements |

[Official Torch installation pairs](https://pytorch.org/get-started/previous-versions/),
[Python compatibility matrix](https://github.com/pytorch/vision#installation),
[xformers compatibility](https://github.com/facebookresearch/xformers/blob/main/CHANGELOG.md),
[CPython 3.12 xformers files](https://pypi.org/project/xformers/0.0.28.post3/),
[CUDA 12.4 xformers wheel index](https://download.pytorch.org/whl/cu124/xformers/),
[MMCV ops-disabled build](https://github.com/open-mmlab/mmcv/blob/v1.7.2/setup.py).

This intentionally differs from Metric3D's historic full training requirements.
It is a **source-informed minimal inference candidate**, not a claim of upstream
certification or successful runtime validation. The resolver, import probes and
real-inference steps below are acceptance gates. Other DA3 transitive versions
must be resolved and frozen; their exact working versions are unknown until that
happens. If resolution fails, stop and revise the candidate explicitly instead
of overriding constraints with `--no-deps` or downgrading the shared environment.

## 7. Exact future commands — not executed in this assessment

All installation, source-fetch and model-fetch commands below require a later
explicit decision to proceed. Run in a **normal host terminal** from the
AI-project repository root. These commands do not change the shared Git branch
or teammates' files; new environments and source downloads go under ignored
`outputs/model_setup/`. Do not copy commands into the current system interpreter.

### A. Create independent environments and constraints

```sh
mkdir -p outputs/model_setup
python3.12 -m venv outputs/model_setup/da3-env
python3.12 -m venv outputs/model_setup/metric3d-env

cat > outputs/model_setup/constraints.txt <<'CONSTRAINTS'
torch==2.5.1
torchvision==0.20.1
numpy==1.26.4
opencv-python==4.11.0.86
xformers==0.0.28.post3
mmcv==1.7.2
timm==0.9.16
matplotlib==3.9.2
CONSTRAINTS

for model_python in outputs/model_setup/da3-env/bin/python outputs/model_setup/metric3d-env/bin/python; do
  "$model_python" -m pip install 'pip==24.3.1' 'setuptools==70.3.0' 'wheel==0.45.1'
  "$model_python" -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu124
  "$model_python" -m pip install -c outputs/model_setup/constraints.txt 'numpy==1.26.4' 'opencv-python==4.11.0.86'
done
```

The official wheel versions include the CUDA local suffix even though the
version constraints omit it. Check `torch.__version__` after installation.
No separate system CUDA reinstall is needed for these prebuilt wheels.

### B. Prepare DA3 package, without fetching weights yet

```sh
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git outputs/model_setup/Depth-Anything-3
git -C outputs/model_setup/Depth-Anything-3 rev-parse HEAD > outputs/model_setup/da3-source-revision.txt

outputs/model_setup/da3-env/bin/python -m pip install 'xformers==0.0.28.post3' --only-binary=:all: --index-url https://download.pytorch.org/whl/cu124
outputs/model_setup/da3-env/bin/python -m pip install --dry-run --report outputs/model_setup/da3-resolution.json -c outputs/model_setup/constraints.txt ./outputs/model_setup/Depth-Anything-3
# Review the resolver output; stop if it requests incompatible core versions.
outputs/model_setup/da3-env/bin/python -m pip install -c outputs/model_setup/constraints.txt ./outputs/model_setup/Depth-Anything-3
outputs/model_setup/da3-env/bin/python -m pip install --no-build-isolation -c outputs/model_setup/constraints.txt -e .
outputs/model_setup/da3-env/bin/python -m pip check
outputs/model_setup/da3-env/bin/python -B -c 'from depth_anything_3.api import DepthAnything3; print("DA3 import OK; pretrained constructor not called")'
```

The source clone is the official basic package; no optional Gaussian/Web UI extras
are selected. A dry run may fetch package metadata or distributions but should
not fetch model weights. Inspect resolver output before actual installation.
There is no validated transitive lock yet; freeze only after successful checks.
Do not silently compile a missing xformers wheel against system CUDA 12.8.

### C. Prepare Metric3D inference imports, without fetching weights yet

```sh
outputs/model_setup/metric3d-env/bin/python -m pip install -c outputs/model_setup/constraints.txt 'timm==0.9.16' 'matplotlib==3.9.2'
MMCV_WITH_OPS=0 outputs/model_setup/metric3d-env/bin/python -m pip install --no-build-isolation -c outputs/model_setup/constraints.txt 'mmcv==1.7.2'
outputs/model_setup/metric3d-env/bin/python -m pip install --no-build-isolation -c outputs/model_setup/constraints.txt -e .
outputs/model_setup/metric3d-env/bin/python -m pip check

git clone --depth 1 https://github.com/YvanYin/Metric3D.git outputs/model_setup/Metric3D
git -C outputs/model_setup/Metric3D rev-parse HEAD > outputs/model_setup/metric3d-source-revision.txt
PYTHONPATH="$PWD/outputs/model_setup/Metric3D" outputs/model_setup/metric3d-env/bin/python -B -c 'from mmcv.utils import Config, collect_env; from mono.model.monodepth_model import get_configured_monodepth_model; import mono.model.backbones, mono.model.decode_heads; print("Metric3D inference imports OK; pretrained constructor not called")'
```

Do not install `mmcv-full` or mmcv 2.x in place of this package. This legacy
source package may still reveal Python/tooling issues at install/import time;
those have not been ruled out by this assessment. The test deliberately imports
the complete relevant model modules rather than only `src.depth_models.metric3d`.
Training/annotation packages from the full requirements file can be added later
if actually needed, under the same constraints. No mock network is involved in
these import checks.

The inspection clone does not redirect the wrapper's Hub loader: the unchanged
wrapper will later fetch/cache its own Hub source. Record/compare that cached
source too. Exact source pinning/local-only initialization is not exposed by
the current wrapper/CLI and remains a reproducibility limitation.

### D. Verify CUDA and core tests before downloading checkpoints

```sh
for model_python in outputs/model_setup/da3-env/bin/python outputs/model_setup/metric3d-env/bin/python; do
  "$model_python" -B -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())'
  "$model_python" -B -m unittest discover -s tests -v
done
```

Existing expected baseline is 64 tests; failures in a new environment must be
investigated. Some tests still use fake model backends by design; they do not
replace the next real-model check. If CUDA is unavailable only in the Codex
sandbox, use the host terminal instead of changing drivers.

### E. Select actual input and matching camera intrinsics

Provide real paths before any model constructor runs:

```sh
: "${REPLAY_VIDEO:?Set to a short local real recording}"
: "${CAMERA_CALIBRATION:?Set to matching camera size/K/dist JSON}"
export REPLAY_VIDEO CAMERA_CALIBRATION

outputs/model_setup/metric3d-env/bin/python -B - <<'PY'
import os
from pathlib import Path
from tello_nav.replay.video import VideoReader
from tello_nav.replay.model_provider import CameraCalibration
frame = next(iter(VideoReader(os.environ['REPLAY_VIDEO'])))
cal = CameraCalibration(Path(os.environ['CAMERA_CALIBRATION']))
print('shape:', frame.rgb.shape, 'focal_px:', cal.focal_for(frame.rgb.shape[1], frame.rgb.shape[0]))
PY
```

Export these variables in the shell so the Python snippet can read them. The
only repository calibration is iPhone-labeled; it must not become Tello
calibration. Matching aspect/orientation alone does not prove matching camera,
crop, distortion or optical settings. The project has no real source recordings
or full cached model maps. Do not use a plot PNG as a calibrated source image.

The adapter expects camera optical-axis Z in meters, aligns HxW, and masks invalid
and Metric3D-saturated depths. It uses mean fx/fy and no undistortion. Accurate
intrinsics/scaling must be verified against the actual pipeline. If a technical
shape-only smoke test uses provisional intrinsics, mark the calibration/scenario
as provisional and do not present inferred values as calibrated distances or
safe-control evidence. Tello metric accuracy stays blocked until Tello calibration
and independent distance checks exist.

### F. Explicitly approved checkpoint loading and first replay

Only after environment/import checks and **separate approval to download weights
and external Hub code**, execute each model in its own process:

```sh
outputs/model_setup/metric3d-env/bin/python -B -m tello_nav --video "$REPLAY_VIDEO" --depth-model metric3d --calibration "$CAMERA_CALIBRATION" --device cuda --allow-model-download --mock-drone --scenario-id metric3d-real-smoke --output outputs/model_setup/metric3d-real.json

outputs/model_setup/da3-env/bin/python -B -m tello_nav --video "$REPLAY_VIDEO" --depth-model da3 --calibration "$CAMERA_CALIBRATION" --device cuda --allow-model-download --mock-drone --scenario-id da3-real-smoke --output outputs/model_setup/da3-real.json
```

These flags are supported by the existing CLI. `--device` otherwise defaults to
CPU, so explicitly select CUDA for the unchanged Metric3D implementation. If
available, add `--timestamps "$FRAME_TIMESTAMPS"` pointing to the synchronized
CSV for this video. Default offline timing tests interface behavior, not realtime
performance. Later `--mode realtime` includes processing delay and may correctly
STOP/reject stale commands.

Constructors handle their official cache layouts, including DA3 config and Hub
code. The current CLI offers no `--checkpoint`, no guaranteed-offline mode, and
no model-revision flag. `--allow-model-download` allows network retrieval even
if some cache entries already exist. Do not imply manually copying a weight file
into an arbitrary directory makes these commands offline.

Inspect output and optionally visualize:

```sh
python -m json.tool outputs/model_setup/metric3d-real.json
python -m json.tool outputs/model_setup/da3-real.json

# GUI-capable host only; repeats actual inference and preserves separate output.
outputs/model_setup/da3-env/bin/python -B -m tello_nav --video "$REPLAY_VIDEO" --depth-model da3 --calibration "$CAMERA_CALIBRATION" --device cuda --allow-model-download --mock-drone --visualize --scenario-id da3-real-view --output outputs/model_setup/da3-real-view.json

outputs/model_setup/da3-env/bin/python -m pip freeze > outputs/model_setup/da3-installed.txt
outputs/model_setup/metric3d-env/bin/python -m pip freeze > outputs/model_setup/metric3d-installed.txt
```

A successful command exit/log write is not enough: inspect every record's `error`,
sector coverage, requested command and `mock_execution.accepted/effective_command`.
Then inspect the actual model output array for HxW alignment, finite-positive
coverage and min/percentiles/max. Measure peak host RAM/GPU allocation and
inference latency on the actual scene; save checkpoint hash/source revision and
camera/input provenance. Logged nominal meter values alone are not an accuracy
validation. No mock adapter action can move a physical drone.

## 8. Decision and next steps

1. **No system-driver repair is indicated:** host NVIDIA/Torch queries work.
   The earlier CUDA-negative observation was sandbox-specific.
2. Free RAM and run one model at a time. Start with Metric3D ViT-Small on host
   CUDA; then test DA3METRIC-LARGE independently, keeping original scaling.
3. Approve preparation of the two isolated candidate environments, run resolver
   and transitive-import checks, and stop on incompatible dependencies. None of
   those installations has been performed.
4. Provide real source RGB and camera-matched calibration. Tello accuracy work
   remains blocked independently of successful network installation.
5. Approve the official pretrained downloads, verify hashes, then execute the
   exact existing-wrapper replay commands and inspect outputs/resources.
6. Freeze a **measured working** lock and checkpoint/code revisions only after
   real inference succeeds. At present both real-model inference and VRAM fit
   remain unverified; no source-grounded assessment can promise exact FPS or
   a fully solved dependency lock without that validation.

No flight experiments, poster work, physical drone operations, package
installation or model downloads were performed during this assessment.
