"""GreenRoute: measure LLM inference energy per completed task at matched quality."""

from greenroute.env import load_env

__version__ = "0.0.1"

load_env()  # .env at the repo root, if present; shell variables take precedence
