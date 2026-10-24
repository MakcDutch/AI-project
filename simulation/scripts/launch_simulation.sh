#!/usr/bin/env bash
# Starts only local PX4 SITL; no physical transport support.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash simulation/scripts/launch_simulation.sh /absolute/PX4-Autopilot' >&2
  exit 2
fi
PX4_ROOT=$(realpath "$1")
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ -x "$PX4_ROOT/build/px4_sitl_default/bin/px4" ]] || { echo 'Build PX4 SITL first; see GAZEBO_PX4_SIMULATION.md' >&2; exit 1; }
[[ -f "$PX4_ROOT/ROMFS/px4fmu_common/init.d-posix/airframes/4002_gz_x500_depth" ]]
[[ -f "$PX4_ROOT/Tools/simulation/gz/models/x500_depth/model.sdf" ]]
command -v gz >/dev/null
command -v setsid >/dev/null
export GZ_DISTRO=harmonic GZ_IP=127.0.0.1
export GZ_PARTITION="${GZ_PARTITION:-ai_project_px4}"
# PX4 v1.16 generated environment script assumes these exist under nounset.
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH-}"
# Load PX4 plugin paths without copying the custom world into the PX4 tree.
source "$PX4_ROOT/build/px4_sitl_default/rootfs/gz_env.sh"
export GZ_SIM_RESOURCE_PATH="$PROJECT_ROOT/simulation/worlds:$GZ_SIM_RESOURCE_PATH"
export PX4_GZ_WORLD=obstacle_world PX4_GZ_STANDALONE=1
export PX4_GZ_MODEL_POSE='0,0,0,0,0,0'
# Empty HEADLESS opens the GUI; default uses EGL without an X server.
export HEADLESS="${HEADLESS-1}"
GZ_JOB='' PX4_JOB=''
cleanup() {
  trap - EXIT INT TERM
  for job in "$PX4_JOB" "$GZ_JOB"; do
    if [[ -n "$job" ]]; then kill -TERM -- "-$job" 2>/dev/null || true; fi
  done
  wait || true
}
trap cleanup EXIT INT TERM
if [[ -n "$HEADLESS" ]]; then
  setsid gz sim -s -r --headless-rendering "$PROJECT_ROOT/simulation/worlds/obstacle_world.sdf" &
else
  setsid gz sim -r "$PROJECT_ROOT/simulation/worlds/obstacle_world.sdf" &
fi
GZ_JOB=$!
cd "$PX4_ROOT"
setsid make -j2 px4_sitl gz_x500_depth <&0 &
PX4_JOB=$!
wait "$PX4_JOB"
