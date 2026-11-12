#!/usr/bin/env bash
# Set up a rented Linux GPU machine (NVIDIA driver already installed) to run
# experiments/run_local.py. Run from the repository root:
#
#     bash scripts/setup_gpu_host.sh
#     source .venv-gpu/bin/activate
#     python experiments/run_local.py phase1 --suites gptoss_pilot
#
# Checks first that the GPU's energy counter is readable, since every result needs it.
set -euo pipefail

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: this machine has no NVIDIA driver" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,power.limit,power.default_limit,memory.total --format=csv

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv-gpu
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
python3 -c "exec(open('greenroute/gpu_requirements.py').read()); print('\n'.join(GPU_REQUIREMENTS))" > /tmp/greenroute-gpu-requirements.txt
uv pip install -r /tmp/greenroute-gpu-requirements.txt modal pytest
uv pip install -e .

python - <<'PY'
import time
import pynvml
pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    time.sleep(1.0)
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
    default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000
    print(f"GPU {i}: energy counter OK ({(e1 - e0) / 1000:.0f} W idle), power limit {limit:.0f} W (default {default:.0f} W)")
    if limit < default:
        print(f"  WARNING: GPU {i} is power-capped below its default; its energy numbers are not comparable")
PY
pytest -q
echo "Ready. Activate with: source .venv-gpu/bin/activate"
