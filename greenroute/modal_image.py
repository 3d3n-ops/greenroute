"""Container image shared by the Modal experiments."""

import modal

from greenroute.bench import VLLM_VERSION


def gpu_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        # vLLM 0.10.x targets transformers 4.x; transformers 5 removed APIs it calls.
        .pip_install(
            f"vllm=={VLLM_VERSION}",
            "transformers>=4.55,<4.57",
            "huggingface-hub<1.0",
            "datasets>=3,<4.2",
            "nvidia-ml-py",
        )
        .add_local_python_source("greenroute")
    )
