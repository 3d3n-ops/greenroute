"""Load settings from a .env file at the repository root.

    cp .env.example .env    # then fill in what you need

Variables already set in the shell win over the file, so a one-off
`GREENROUTE_GPU=A100-80GB modal run ...` still works. Values are never printed.
No dependency: supports KEY=VALUE lines, `export KEY=VALUE`, # comments and
single or double quotes. Inside a Modal container there is no .env, so this does nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif value.startswith("#"):
            value = ""  # KEY=   # comment
        else:
            value = value.split(" #", 1)[0].rstrip()  # trailing comment on an unquoted value
        out[key] = value
    return out


def load_env(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Set variables from the file; returns the names set (not their values)."""
    p = Path(path) if path else ENV_FILE
    if not p.is_file():
        return []
    set_names = []
    for key, value in parse(p.read_text()).items():
        if value == "" or (key in os.environ and not override):
            continue  # an empty value in .env means "not set", so it never hides a default
        os.environ[key] = value
        set_names.append(key)
    return set_names
