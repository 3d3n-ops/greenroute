"""Packages for GPU runs: one list for the Modal image and for rented machines.

Plain Python with no imports, so scripts/setup_gpu_host.sh can read it before
anything is installed. Keep vLLM in step with greenroute/bench.py.
"""

GPU_REQUIREMENTS = [
    "vllm==0.10.1.1",
    # vLLM 0.10.x targets transformers 4.x; transformers 5 removed APIs it calls.
    "transformers>=4.55,<4.57",
    "huggingface-hub<1.0",
    "datasets>=3,<4.2",
    "nvidia-ml-py",
]
