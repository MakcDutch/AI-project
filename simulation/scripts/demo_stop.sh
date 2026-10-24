#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv-simulation/bin/python" -B -m tello_nav.simulation.demo stop "$@"
