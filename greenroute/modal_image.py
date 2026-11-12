"""Container image shared by the Modal experiments."""

import modal

from greenroute.bench import VLLM_VERSION
from greenroute.gpu_requirements import GPU_REQUIREMENTS

assert f"vllm=={VLLM_VERSION}" in GPU_REQUIREMENTS, "pin the same vLLM in greenroute/gpu_requirements.py and bench.py"


def gpu_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*GPU_REQUIREMENTS)
        .add_local_python_source("greenroute")
    )
