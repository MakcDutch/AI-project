# Verified PX4 X500 + Gazebo Harmonic closed-loop simulation

Verified on 2026-10-24. **All three actual flight experiments passed.** These
results use real PX4 SITL, Gazebo physics, the X500 depth sensor, MAVSDK Offboard
commands and the existing Reactive Controller. No mock flight backend or neural
model was substituted. Everything remained virtual; no Tello commands,
physical-flight operations, commits or pushes were performed.

## Interactive recording demonstration (manual flight start)

From the AI-project directory, use these four separate commands. They select
`.venv-simulation/bin/python` automatically; activating a venv is unnecessary.
The existing built PX4 checkout defaults to `/tmp/ai-project-PX4-Autopilot`.
For a different existing build, set `PX4_ROOT=/absolute/PX4-Autopilot` **before
prepare**. The selected path is recorded and reused by the other commands.

```bash
# 1. Prepare only: actual camera/telemetry check, unarmed X500 on ground.
./simulation/scripts/demo_prepare.sh

# 2. Open the GUI. Confirm x500_depth_0 and the orange wall are visible.
./simulation/scripts/demo_gui.sh

# Enable screen recording now. No flight occurs while you wait.
# 3. Explicitly start: takeoff -> stable hover -> approach -> stop -> land/disarm.
./simulation/scripts/demo_start.sh

# 4. When finished recording, stop this demo's GUI, PX4 and Gazebo server.
./simulation/scripts/demo_stop.sh
```

Prepare returns while a detached supervisor keeps the existing launcher and
its PX4 console PTY alive. Physics runs idle at the unchanged **0.25× profile**;
the world is not paused, and preparation never arms or takes off. Readiness
uses the existing real `--smoke` runner, checks depth/clock/model pose/contact
advertisement/MAVSDK communication, then reads PX4 commander status to confirm
**Disarmed**. Nothing schedules a flight after readiness. Repeating prepare on
an unused live spawn is harmless; a used spawn requires stop then prepare.

The GUI receives the exact recorded partition, loopback transport address and
PX4 model/resource paths. Its fixed initial viewpoint shows the wall and X500;
mouse camera navigation remains available. Pause/reset and world-editing
toolbars are omitted to reduce accidental interference with physics. GUI and
server remain alive after a successful or failed experiment. Closing the GUI
does not stop the server; rerun `demo_gui.sh` to reopen it. Closing the terminal
that prepared the demo does not stop the server. `demo_start.sh` occupies its
terminal until landing, then prints the actual result and generates the plot.
A flight consumes the spawn even if it fails, avoiding accidental reuse of a
displaced vehicle. Use stop/prepare for each new demonstration.

Ctrl+C in the start terminal requests the existing runner's cancellation and
landing path. If stop is requested during flight, it first signals that runner
and waits for its cleanup before stopping physics; if cleanup has not finished
within 100 wall seconds, stop fails and leaves physics running for inspection.
This active-flight interruption path is covered by existing runner safety
logic but was **not physically retested during this demo-workflow change**.
Normal completed-flight cleanup was verified against the actual processes.
Process records include Linux PID start ticks to avoid signalling reused PIDs.
Only the recorded GUI group, supervisor and launcher's own children are stopped;
there is no `pkill`, `killall` or blanket Gazebo/PX4 process termination.
Preparation refuses another PX4 instance because SITL uses fixed instance/ports.
Do not delete the process-state files while the demonstration is running.

### Why plain `gz sim -g` failed

At inspection there was **no live Gazebo/PX4 server**. The previous foreground
launcher explicitly cleaned up its children on exit. In addition, that launcher
exported `GZ_PARTITION=ai_project_px4` only inside itself, while the desktop
shell had no `GZ_PARTITION`, `GZ_IP` or PX4 `GZ_SIM_RESOURCE_PATH`. A separately
invoked GUI therefore had neither the matching discovery partition nor the
model paths. The demo now records a fresh `ai_project_demo_<id>` partition and
reuses it for every component, without modifying the user's shell environment.
The legacy launcher's default partition remains unchanged when no override is
provided. `GZ_IP=127.0.0.1` keeps transport local.

The current desktop is X11 (`DISPLAY=:1`); the GUI selects Qt's `xcb` backend
there and otherwise retains the user's display environment. Run the GUI from a
graphical desktop terminal. Its startup is checked for immediate failure, but
process liveness alone does not prove scene visibility: visually confirm the
wall and X500 before starting. GUI logs are retained. Actual desktop captures
confirmed successful rendering and model loading on this machine. EGL NVIDIA
`failed to create dri2 screen` messages and Qt QML warnings occurred but did
**not** prevent that observed rendering. No driver reinstall or software-render
forcing was necessary. Do not force `LIBGL_ALWAYS_SOFTWARE=1`: an earlier
headless diagnostic failed with that override. The known warning alone is not
a reason to change the verified renderer/profile.

### Workflow verification on 2026-10-24

A real prepare -> GUI -> explicit start -> stop cycle was performed, with the
GUI open throughout flight and still open after landing. Desktop captures
show the actual wall and X500 before flight and at its final position. The
Gazebo pose was also read independently during flight at `(4.069,-0.359,1.278)`
meters, with `collision=false` and no sensor fault.

The GUI-enabled wall experiment **passed**: minimum navigation clearance
**1.403821 m** (all-phase sampled minimum **1.403823 m**), measured stopping travel **0.269234 m**, detection-to-stop
**1.004 simulation seconds**, peak measured speed **0.471908 m/s**. The applied
forward setpoint was **0.45 m/s**, followed by zeros for braking/stopping; the
rendered plot confirms physical deceleration. It recorded **zero collisions**,
no safety events or measurement gaps, then landed/disarmed. A separate read-only
`px4-commander status` confirmed **Disarmed**. Repeated start and prepare on the
used spawn were rejected without another flight. Cleanup removed the recorded
supervisor, launcher, GUI and PX4 processes; a deliberately unrelated process
survived. The 0.25× world and controller/adapter safety logic were preserved.
Screen recording adds load that this check does not reproduce; existing
freshness/watchdog gates remain active and must not be loosened to mask load.

Evidence for that completed cycle is retained under
`outputs/simulation/demo_history/ai_project_demo_c298e98a4945/`, including
`wall.json`, `wall.png`, `flight.log`, `simulator.log`, `gui.log`,
`prepared_desktop.png`, `verified_flight_gui.png`, `cleanup_verification.json`
and the standard test log. A fresh prepared spawn is left afterwards; **no
flight is initiated on that final spawn**.

Current state and logs are in `outputs/simulation/demo/`: `config.json` records
the checkout, partition, owned PX4 identity and whether flight was requested;
`ready.json`, `ground_status.log` and `ready.log` document nonarming readiness.
`wall.json`, `flight.log` and `wall.png` appear only after manual start. A new
preparation archives the previous session under `demo_history/<partition>/`
without deleting its flight evidence. Existing `outputs/simulation/wall.json`
from the earlier three-experiment campaign is preserved separately.

```bash
python -m json.tool outputs/simulation/demo/ready.json
cat outputs/simulation/demo/ground_status.log
# During/after a manually started flight:
tail -f outputs/simulation/demo/flight.log
python -m json.tool outputs/simulation/demo/wall.json
xdg-open outputs/simulation/demo/wall.png
# Standard regression suite (live integration intentionally skipped):
.venv-simulation/bin/python -B -m unittest discover -s tests -v
```

Eight focused lifecycle tests were added for ownership, recycled PID rejection,
nonarming preparation, distinct smoke/flight commands, duplicate-flight guards,
missing-session rejection and the demo GUI layout. The simulator-free run was
**127 tests run: 126 passed, 1 live integration test skipped**, in 10.874 s.
The real flight above is separate evidence, not inferred from those unit tests.
After the final fresh preparation, the complete suite was rerun with
`AI_PROJECT_SITL_SMOKE=1`, the recorded `PX4_ROOT`/`PX4_PID`, and the demo's
partition/environment: **127 passed in 9.962 s, zero skipped or failed**.
`outputs/simulation/demo/tests_with_sitl.log` retains that output. This live
regression check never arms. Final read-only process/pose/commander checks are
retained in `outputs/simulation/demo/final_prepared_verification.json`.

## Actual results

The reproducible profile runs physics at **0.25× real time**. Distances are
meters, speeds are m/s, and detection-to-stop durations below are **simulation
seconds**. Minimum clearance subtracts a conservative 0.5 m vehicle envelope
from the measured distance between the model origin and the wall front face.
It is not an exact collision-mesh surface distance.

| Experiment | Outcome | Minimum clearance, entire recorded flight | Navigation minimum clearance | Measured stopping travel | Detection to sustained stop | Peak measured speed | Wall contacts | Final vehicle state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: clear path | PASS | 4.021671 | 4.438953 | N/A: no obstacle/fault stop trigger | N/A | 0.467696 | 0 | landed/disarmed |
| B: wall | PASS | 1.332426 | 1.333802 | 0.278845 | 0.984 s | 0.472189 | 0 | landed/disarmed |
| C: interrupted depth | PASS | 3.725679 | 3.725678 | 0.271081 | 1.036 s | 0.460409 | 0 | landed/disarmed |

Entire-flight clearance uses Gazebo poses sampled alongside telemetry through
takeoff, navigation and landing. Navigation records sample independently, so
tiny differences between their minima are normal. The runtime summary's
`minimum_clearance_m` is the **navigation** minimum; the all-phase minimum is in
`flight_verification.json`. A coasts farther during its controlled landing than
its navigation endpoint, explaining its lower full-flight clearance.

A moved **1.005657 m during navigation**, entered FORWARD, and recorded no safety
events. All three flights requested a maximum forward velocity of **0.45 m/s**.
Their sampled sensor-update rates were approximately **30 Hz in simulation
seconds**, or **7.5 Hz in wall seconds**. Navigation took 11.731 wall seconds in
A, 51.372 in B and 32.315 in C, separate from takeoff and landing.

B established the complete chain:

1. Center-sector depth decreased from **5.424754 m to 1.701251 m**. Initial pitch
   and ground visibility cause a short initial depth dip/recovery; the approach
   signal is not strictly monotonic pixel by pixel.
2. After measured forward flight, the controller entered BRAKE at simulation
   time **33.444 s** (8.832 s into navigation).
3. The adapter requested zero body velocity. PX4's own `trajectory_setpoint`
   ULog contains forward-phase horizontal magnitude **0.450000 m/s**, then zero.
   The first logged zero is at **33.552 s**, an observed upper bound of
   **0.108 simulation seconds** from detection, limited by log sampling.
4. Gazebo ground-truth speed was **0.473637 m/s** at detection and at most
   **0.015629 m/s** during the final navigation second. This independently
   confirms physical deceleration, rather than inferring rest from a setpoint.
5. The measured pose path from detection to the start of a confirmed low-speed
   interval was **0.278845 m**. Speed remained below **0.08 m/s for a continuous
   simulation second** to confirm stopping. No wall contacts or safety overrides
   occurred, and controlled landing/disarming completed.

C deliberately stopped **consuming** the real camera stream after 4 simulation
seconds while measured forward speed exceeded 0.1 m/s. It neither froze physics
nor injected invented depth. Frame **549**, with original capture stamp
**29.400 s**, remained frozen in the consumer. Interruption was logged at
**29.424 s**; timeout inhibition and zero motion were logged at **29.488 s**,
**0.259518 wall seconds** after interruption. Every subsequent applied setpoint
was zero. The command-age deadline fired first; its age includes time since the
last consumed frame, before the explicit interruption marker. PX4 recorded 19
post-inhibition zero-setpoint samples in the audited interval. Ground-truth
speed fell from **0.457848 m/s** at inhibition to at most **0.017958 m/s** in the
final navigation second. The expected timeout event is counted as successful
fault handling, not a failure of C.

## Evidence files and inspection

Runtime evidence is under `outputs/simulation/` (Git-ignored):

- `clear.json`, `wall.json`, `stale.json`: actual navigation records, lifecycle
  events, all-phase telemetry/Gazebo pose history, contacts and summaries.
- `clear.ulg`, `wall.ulg`, `stale.ulg`: preserved onboard PX4 logs, including
  `trajectory_setpoint`, estimated velocity and independent simulator truth.
- `clear.png`, `wall.png`, `stale.png`: rendered and visually inspected plots of
  distance, requested/applied/measured velocity, state and trajectory.
- `wall_audit.json`, `stale_audit.json`: independent checks of flight records
  against the actual onboard setpoints and physics velocities.
- `flight_verification.json`: consolidated results, full-flight clearance and
  SHA-256 hashes of each flight log, ULog and plot.
- `smoke.json`: persistent real nonarming communication evidence, including
  depth dimensions, original sensor time, world pose and MAVSDK telemetry.
- `tests_with_sitl.log`: complete regression output with actual live test enabled.
- `contact_probe.json`: successful isolated physical collision positive control.
- `*_simulator.log`, `*_runner.log`: startup, takeoff, failure/recovery and CLI output.

```bash
python -m json.tool outputs/simulation/flight_verification.json
python -m json.tool outputs/simulation/wall_audit.json
python -m json.tool outputs/simulation/stale_audit.json
python -B -m tello_nav.simulation.plots outputs/simulation/wall.json --output outputs/simulation/wall.png
python -B -m tello_nav.simulation.plots outputs/simulation/clear.json --output outputs/simulation/clear.png
python -B -m tello_nav.simulation.plots outputs/simulation/stale.json --output outputs/simulation/stale.png
```

The plot CLI consumes JSON. CSV output is also supported by the flight runner,
using the existing logging convention and a metadata sidecar. Plot rendering
rejects an empty/nonflight log. Files in `/tmp`, including the external PX4
checkout, may be removed at reboot; the copied ULogs remain in project outputs.

## BRAKE/STOP alternation: investigated, controller preserved

B contains **25 state entries/transitions**, including repeated BRAKE/STOP
alternation after obstacle detection. This follows the existing state machine:
blocked STOP enters BRAKE to consider avoidance; after its 0.3 s braking dwell,
`allow_lateral=false` returns STOP because side clearance is unobserved. A still
blocked next frame starts the same consideration again.

This is state-label churn in this configuration, **not renewed forward flight**:

- zero FORWARD states after first BRAKE;
- zero nonzero applied STOP/BRAKE samples;
- all 19 audited PX4 setpoint samples from detection +0.25 s through navigation
  end have exactly zero horizontal velocity;
- simulator ground-truth speed is below 0.016 m/s in the final second;
- no wall contacts, no safety override, and more than 1.33 m conservative clearance.

It did not impair stopping safety in the measured run. The Reactive Controller
was **not changed**. The alternation may complicate state-only analytics; use
actual applied setpoints and velocity, not state labels alone. This result does
not establish safety for arbitrary speeds, lateral avoidance or new worlds.

## Architecture, units and clocks

```text
Gazebo depth camera -> GazeboCamera / DepthFrame
 -> aligned float32 H×W optical-axis Z meters + original capture time
 -> existing ReactiveController and sector extraction
 -> existing ControlCommand (RC units)
 -> explicit SI mapping -> independent SafetyMonitor
 -> MAVSDK body velocity stream -> PX4 Offboard -> real Gazebo X500 dynamics

Gazebo clock / model pose / wall contacts + PX4 telemetry -> safety and logs
```

Simulation is isolated in `tello_nav/simulation/`. Existing controller, replay,
DA3/Metric3D wrappers and Tello adapters remain unchanged. It reuses existing
sector extraction, `ControlCommand` and `replay.logging.write_results`; no
controller state machine was duplicated. The replay CLI remains unchanged.

The pinned OakD-Lite sensor is 640×480 R_FLOAT32, horizontal FOV 1.274 rad,
near clip 0.2 m, far clip 19.1 m and nominal rate 30 Hz. Its camera-link offset
is `(0.01233,-0.03,0.01878)` and the X500 camera attachment is
`(0.12,0.03,0.242)`. Ogre2 exports camera-forward axis distance, equivalent to
optical Z, **not Euclidean point-cloud range**. The decoder verifies dimensions,
encoding, stride, byte count and timestamp, and converts nonfinite/nonpositive
values to unknown NaN. It never fills holes or invents free space. Dimension
changes, bad messages, reordered stamps and clock resets latch a fault.

Discovery checks advertised `gz.msgs.Image` types, excludes point clouds/info,
and distinguishes the RGB sensor path from `x500_depth_0` in the model name.
Exactly one depth Image stream must remain. The actual topic is `/depth_camera`.

Clock domains remain explicit:

- depth capture stamps, controller clock and Gazebo pose time: simulation seconds;
- receipt freshness, command deadlines and watchdog: monotonic wall seconds;
- telemetry additionally records the observed world clock at receipt for the
  controller's simulation-domain freshness gate. This is a **receipt** time,
  not an invented telemetry capture stamp. MAVSDK's velocity interface does not
  expose the original MAVLink LOCAL_POSITION_NED timestamp;
- onboard ULog timestamps provide independent PX4 simulation-time evidence.

World clock minus depth capture time is an observed simulation-age measurement;
it includes sampling/transport skew and is not a synchronized hardware latency.
The all-phase telemetry history preserves both receipt domains and actual
Gazebo poses. A later neural provider must preserve original RGB capture time
and produce aligned metric Z through the existing shared adapter. Neural
inference is not part of these experiments. Simulated depth uses no iPhone or
Tello calibration; adding RGB neural depth requires the simulated RGB camera's
own intrinsics and RGB/depth registration, because resolutions/FOVs differ.

Mapping at RC limit 20:

- forward = FB/20 × 0.6 m/s; right = LR/20 × 0.6 m/s;
- down = −UD/20 × 0.3 m/s; clockwise yaw = yaw/20 × 15 deg/s;
- horizontal magnitude is capped at 0.6 m/s;
- FB=15 requests **0.45 m/s**; STOP/BRAKE request zero in every channel.

These are selected simulation gains, not a Tello speed calibration. MAVSDK body
commands use FRD; its local telemetry uses NED. Gazebo world is ENU:
`(north,east,down)=(y,x,-z)`. Spawn yaw=0 points toward world +X/east and the wall;
PX4 NED yaw is checked near 90 degrees before navigation. The velocity plots
project NED velocity onto measured heading; exact simulator speed is checked
separately through ULog ground truth. Pose quaternions and Euler telemetry are
also recorded.

## Environment and completed build

Verified versions and locations:

| Dependency | Actual status |
| --- | --- |
| Ubuntu / Python | Ubuntu 24.04, Python 3.12.3; simulation requires Python 3.11+ |
| Gazebo | Harmonic Sim 8.15.0, Transport 13.6.0, Msgs 10.4.0; Python bindings installed |
| Native OpenCV / pkg-config | OpenCV 4.6.0, pkg-config 1.8.1; detected successfully by CMake |
| PX4 | v1.16.0, `6ea3539157ca358c70a515878b77077af7d4611d` |
| External source/build | `/tmp/ai-project-PX4-Autopilot`, built `px4_sitl_default` successfully |
| Gazebo model submodule | `e05f4312d3f28aa621157610584a4870406cb6d3` |
| MAVSDK | 3.17.4 in `.venv-simulation`; actual gRPC/Offboard interfaces verified |
| Build tools | GCC/G++ 13.3, CMake, Ninja, Eigen present; two build jobs used |
| Python build tools | Upstream PX4 requirements installed in simulation venv; includes empy 3.3.4, kconfiglib 14.1.0, toml 0.10.2, pyros-genmsg 0.5.8 |
| ULog audit | pyulog 1.3.0; pymavlink 2.4.50 also available for read-only diagnostics |
| ROS 2 / neural weights | Not required or installed for this integration |

The previous OpenCV blocker is resolved. The next actual build failure was
missing `mavlink/pymavlink/tools/mavgen.py`; initializing the nested MAVLink
submodule fixed it. The complete 1092-step build passed without PX4 source
patches. `gz_x500_depth` was then built/launched successfully. Missing GStreamer
development packages emit an optional plugin warning; that video plugin is
not needed for native depth transport and did not block building or flying.

Reproduce/resume using the existing external checkout:

```bash
source .venv-simulation/bin/activate
export PX4_ROOT=/tmp/ai-project-PX4-Autopilot
pkg-config --modversion opencv4
python -m pip install -r "$PX4_ROOT/Tools/setup/requirements.txt"
git -C "$PX4_ROOT" submodule update --init --recursive --depth 1 src/modules/mavlink/mavlink src/lib/events/libevents src/drivers/gps/devices src/modules/uxrce_dds_client/Micro-XRCE-DDS-Client src/lib/heatshrink/heatshrink
GZ_DISTRO=harmonic PYTHONDONTWRITEBYTECODE=1 make -C "$PX4_ROOT" -j2 px4_sitl_default
python -B -m tello_nav.simulation.preflight --px4-root "$PX4_ROOT"
```

For a fresh environment, create the venv with `--system-site-packages` so the
installed Gazebo Python bindings are visible, then install the optional project
extras and build requirements. These commands are supported by this project:

```bash
python3 -m venv --system-site-packages .venv-simulation
source .venv-simulation/bin/activate
python -m pip install -e '.[simulation]'
python -m pip install -r simulation/requirements-build.txt
```

If the external checkout is gone, use a new/nonexistent destination:

```bash
export PX4_ROOT=/tmp/ai-project-PX4-Autopilot
git clone --depth 1 --branch v1.16.0 --filter=blob:none --no-checkout https://github.com/PX4/PX4-Autopilot.git "$PX4_ROOT"
git -C "$PX4_ROOT" sparse-checkout set --cone src platforms msg cmake ROMFS Tools boards/px4/sitl
git -C "$PX4_ROOT" checkout
```

Then run the requirements/submodule/build commands above; CMake initializes
other build-required submodules as needed. No branch, commit or push is needed.
An administrator must supply native build packages on a fresh host; on this
machine the user already installed `libopencv-dev` and `pkg-config`.

## Verified launch and experiment commands

Run from AI-project. Each flight needs a **fresh simulator ground pose**. The
adapter rejects an already moved/airborne vehicle at startup.

Terminal 1:

```bash
source .venv-simulation/bin/activate
export PX4_ROOT=/tmp/ai-project-PX4-Autopilot
bash simulation/scripts/launch_simulation.sh "$PX4_ROOT"
```

The launcher sources the generated PX4 plugin paths with safe unset-variable
defaults, starts this repository's world, and runs the real target
`make px4_sitl gz_x500_depth` with `PX4_GZ_STANDALONE=1`. It forwards console
stdin explicitly so backgrounding make does not create an EOF/prompt busy loop.
It does not copy or patch the custom world into PX4. The default is headless EGL;
use `HEADLESS= bash simulation/scripts/launch_simulation.sh "$PX4_ROOT"` for GUI.

Terminal 2, also from AI-project:

```bash
source .venv-simulation/bin/activate
export PX4_ROOT=/tmp/ai-project-PX4-Autopilot
export GZ_PARTITION=ai_project_px4 GZ_IP=127.0.0.1
export PX4_PID=$(python -B -m tello_nav.simulation.preflight --px4-root "$PX4_ROOT" --print-pid)
# Actual communication verification; never arms.
python -B -m tello_nav.simulation.runner --simulation-only --px4-root "$PX4_ROOT" --px4-pid "$PX4_PID" --smoke --output outputs/simulation/smoke.json
```

Execute **one** flight per fresh simulator launch, obtaining the new PID after
restarting Terminal 1:

```bash
python -B -m tello_nav.simulation.runner --simulation-only --px4-root "$PX4_ROOT" --px4-pid "$PX4_PID" --experiment clear --output outputs/simulation/clear.json
python -B -m tello_nav.simulation.runner --simulation-only --px4-root "$PX4_ROOT" --px4-pid "$PX4_PID" --experiment wall --output outputs/simulation/wall.json
python -B -m tello_nav.simulation.runner --simulation-only --px4-root "$PX4_ROOT" --px4-pid "$PX4_PID" --experiment stale --output outputs/simulation/stale.json
```

A ends navigation after 1 m measured travel. B/C continue four simulation seconds
after first brake/inhibition to measure deceleration and sustained rest, then
land. Default navigation duration is 25 simulation seconds (`--duration` 1–60),
with a separate 120 wall-second navigation limit. Takeoff/hover is bounded by
35 simulation seconds and 180 wall seconds, and initialization has a 300-second
wall bound. These are separate from the strict ongoing motion freshness checks.

Ctrl+C/SIGTERM in **Terminal 2 first** requests emergency zero/controlled landing.
Wait for `landed_disarmed` / confirmed landing, then stop Terminal 1. Its trap
terminates only its owned PX4/Gazebo process groups. Restart the launcher to
reset physics/model pose and obtain a new PID; controller/safety state is new
per run. Do not reset the world clock during an active experiment.

## Safety and initialization fixes validated during actual execution

Before connecting, require `--simulation-only` and a PID whose `/proc/PID/exe`
exactly matches this checkout's built SITL binary and X500-depth airframe.
Require the expected world clock, depth, initial X500 pose and wall-contact
publisher before arming. MAVSDK accepts only `udpin://127.0.0.1:14540`; no serial
or remote-vehicle URL is exposed. These are simulation guards, not a security
boundary against intentional physical-device forwarding.

The adapter configures the verified local SITL's onboard heartbeat stream with:

```bash
/tmp/ai-project-PX4-Autopilot/build/px4_sitl_default/bin/px4-mavlink stream -u 14580 -s HEARTBEAT -r 5
```

The adapter performs this automatically **after verifying the SITL PID**. This
is a local simulator stream-rate command, not a physical-drone command. At
0.25×, a default 1 Hz simulation heartbeat arrives every four wall seconds and
exceeds MAVSDK's three-second connection timeout. Actual diagnostic traces
showed repeated disconnect/reconnect and home health resetting. Five simulation
Hz yields 1.25 wall Hz and keeps communication stable. Home-position streaming
is requested explicitly as well.

Before arming, the adapter verifies global/home health, writes and reads back
`COM_OF_LOSS_T=0.5` and `COM_OBL_RC_ACT=4` (Land), sets/read-checks the 1.5 m
takeoff altitude, and waits 1.2 **simulation** seconds for Navigator's cached
parameter update. It verifies two simulation seconds of low-speed hover, then
prestreams zero setpoints for 1.25 simulation seconds and enters Offboard.
Zero updates continue while waiting for mode telemetry. The SDK resends cached
setpoints at 20 wall Hz; the control loop also refreshes them regularly.

Independent latched safety checks enforce 0.6 m/s horizontal command magnitude,
0.3 m/s vertical command, 3 m maximum navigation altitude, 8 m travel, >0.65 m
wall clearance, 0.5 wall-second sensor/telemetry freshness and a 0.3 wall-second
command deadline. Clock, pose and attitude freshness, Offboard mode, contact
publisher availability, nonfinite telemetry, wall contacts, manual stop and
measured horizontal overspeed (>configured max +0.2 m/s) also inhibit motion.
There is no automatic forward recovery after an inhibition.

An independent thread terminates only its owned MAVSDK server if the command
loop stops refreshing in Offboard, preventing indefinite cached forward motion;
PX4's configured Offboard-loss landing then applies. This uses the pinned
MAVSDK 3.17.4 server-process interface. Controlled cleanup sends zero and lands;
disarm is requested only after `ON_GROUND`. Failed landing invalidates success.
This is best-effort software supervision, not hard real-time certification.

Initial failed trials were retained, not counted as successful experiments:

| Trial evidence | Actual failure and correction |
| --- | --- |
| `initial_simulator.log` | Unset Gazebo path variable under shell `set -u`; launcher now initializes defaults |
| `startup_simulator.log` / first smoke | RGB topic matched model name `x500_depth_0`; discovery now examines sensor-relative paths; console EOF/prompt spin fixed by explicit stdin forwarding |
| `clear_attempt1.json` | Real-time trial had estimator divergence and used cached 2.5 m takeoff altitude; hover check failed and landing/disarming occurred. The verified profile rate-limits physics to 0.25× and waits for parameter propagation. Scheduling pressure is a suspected cause of divergence, not an isolated/proven root cause |
| `clear_attempt2.json`, `clear_home_failure.json` | Repeated wall-clock heartbeat timeouts prevented home-health readiness; fixed stream rates, without bypassing health gates |
| `clear_hover_timeout.json` | Wall-clock hover timeout expired during slowed takeoff; changed phase waits/dwell to simulation time with separate wall bounds |
| `clear_offboard_entry_failure.json` | Mode-telemetry wait did not refresh zero commands; independent watchdog killed SDK, triggering PX4 landing failsafe; continuous refresh added |

The oldest failed trial predates improved lifecycle logging; its `not_armed`
summary label is not evidence that no takeoff happened. PX4 console/ULog showed
takeoff and landing. Final logs preserve all-phase history to avoid that
ambiguity. Failed trial contact status is not retroactively asserted.

`offboard_entry_watchdog_recovery.json` independently captured actual recovery
following the watchdog failure: Gazebo model on the floor, MAVLink
`landed_state=ON_GROUND`, and heartbeat with the armed bit clear. This verifies
one observed fallback recovery, not a systematic failsafe campaign.

## Collision positive control and automated tests

The corrected wall sensor consumes `gz.msgs.Contacts`, whose collisions are
**Entity messages**, not strings. Any contact on this wall sensor latches;
unnamed IDs cannot silently disappear. Final flight logs include the complete
collected contact events, including takeoff and landing periods.

A clean, separate Gazebo scene physically intersected a dynamic cube with the
same wall. Two actual contact records identified the unique probe and wall;
`contact_probe.json` reports success. This verifies transport/parsing/latching,
not aircraft crash resistance. Run this only in its isolated partition:

```bash
python -B -m tello_nav.simulation.diagnostics --make-probe-world /tmp/ai_project_contact_probe.sdf
GZ_PARTITION=ai_project_contact_probe GZ_IP=127.0.0.1 gz sim -s -r --headless-rendering /tmp/ai_project_contact_probe.sdf
# In another terminal, from AI-project:
GZ_PARTITION=ai_project_contact_probe GZ_IP=127.0.0.1 python -B -m tello_nav.simulation.contact_probe --output outputs/simulation/contact_probe.json
```

Use a fresh world or choose a new generated-world path if it already exists.
World creation refuses overwriting a file. Stop that Gazebo process afterwards.
The probe uses Gazebo's native service client so the synchronous Python request
binding cannot starve subscribed contact callbacks while holding Python's GIL.

**Earlier three-experiment campaign: 119 tests passed in 6.558 s; zero skipped or failed.** This
included the real nonarming Gazebo/PX4 integration test against a fresh simulator.
New regressions cover Entity contact parsing, delayed Offboard-mode telemetry
without a watchdog trip, and separation of telemetry clock domains. Existing
controller, replay, prediction exchange, model-interface and Tello tests passed.

```bash
# Simulator-free normal suite; real integration test is deliberately skipped.
python -B -m unittest discover -s tests -v
# All tests including actual communication: start a fresh simulator first,
# set PX4_ROOT/PX4_PID and Gazebo environment as above, then:
AI_PROJECT_SITL_SMOKE=1 python -B -m unittest discover -s tests -v
```

The normal unit suite does not require Gazebo or MAVSDK imports at runtime;
optional plotting tests skip if Matplotlib is absent. The live test requires
`PX4_ROOT`, `PX4_PID`, matching partition and explicit opt-in, and never arms.
Actual flight experiments are separately verified above; unit tests are not
being presented as flight evidence.

## Limits and troubleshooting

- **X500 only**: this is not a validated Tello dynamics model, a learned-depth
  accuracy result, or permission for physical flight. DA3/Metric3D were not run.
- **Verified 0.25× profile**: faster profiles, higher speeds, different camera
  rates/worlds and lateral avoidance need fresh flight verification. The
  configured 0.5 m/s² braking parameter remains an initial tuning assumption;
  actual response is measured in these logs, not inferred from that parameter.
- **GPU/EGL**: headless runs succeeded on this host despite Mesa device warnings.
  Forcing `LIBGL_ALWAYS_SOFTWARE=1` previously crashed EGL/Ogre2. Do not infer a
  driver failure or replace drivers from those warnings. Try GUI mode if needed.
- **Restricted sandbox**: real Gazebo Transport/PX4 needs local sockets and
  device access; actual runs used approved execution outside the tool sandbox.
  Normal local shell execution is the intended reproduction environment.
- **GStreamer warning**: optional plugin unavailable without development packages;
  native depth transport and these actual experiments work without it.
- **State churn**: BRAKE/STOP alternation is explained above. Judge motion from
  applied setpoints, actual velocity and contacts, not the state label alone.
- **No depth**: check matching partition and `gz topic -l`, then
  `gz topic -i -t /depth_camera`; run exactly one X500 depth stream per partition.
- **MAVSDK v4**: the package name is transitioning to a native API. This code pins
  3.17.4; `mavsdk-grpc` is the continuation of this async API. Upgrade only with
  review of owned-server shutdown and another actual flight verification.
- **Matplotlib**: a mixed system/user install warns that Axes3D is unavailable;
  these four 2D plots rendered and were visually inspected successfully. Set
  `MPLCONFIGDIR=/tmp/ai-project-matplotlib` for a writable cache when needed.
- **ACK 176 warning**: MAVSDK logged an occasional late duplicate mode-command
  acknowledgement. Final runs independently confirmed Offboard mode, setpoint
  reception, physics response, landing and disarming; no success was inferred
  from an acknowledgement warning alone.

No requested experiment remains blocked or unverified in the tested profile.
Broader fault campaigns, actual aircraft contact tests and physical Tello/model
inference validation remain outside these results.

## Official references

- [PX4 v1.16 Gazebo simulation](https://docs.px4.io/v1.16/en/sim_gazebo_gz/)
- [Pinned PX4 Gazebo build targets](https://github.com/PX4/PX4-Autopilot/blob/v1.16.0/src/modules/simulation/gz_bridge/CMakeLists.txt)
- [Pinned world/startup handling](https://github.com/PX4/PX4-Autopilot/blob/v1.16.0/ROMFS/px4fmu_common/init.d-posix/px4-rc.gzsim)
- [Pinned models](https://github.com/PX4/PX4-gazebo-models/tree/e05f4312d3f28aa621157610584a4870406cb6d3/models)
- [Gazebo depth sensor implementation](https://github.com/gazebosim/gz-sensors/blob/gz-sensors8/src/DepthCameraSensor.cc)
- [MAVSDK Offboard streaming](https://mavsdk.mavlink.io/main/en/cpp/guide/offboard.html)
- [MAVSDK telemetry/home handling](https://github.com/mavlink/MAVSDK/blob/v3.17.0/src/mavsdk/plugins/telemetry/telemetry_impl.cpp)
