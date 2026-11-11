"""Which code produced a result."""

from __future__ import annotations

import subprocess

CODE_PATHS = ("greenroute", "experiments", "pyproject.toml")


def git_sha(cwd: str | None = None) -> str:
    """Short commit hash, with '-dirty' appended when the code differs from that commit.

    A result stamped with a clean hash can be rerun from the commit. A '-dirty' one
    was produced by code that is not in the history, so the hash alone is not enough.
    """
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True, cwd=cwd, stderr=subprocess.DEVNULL)

    try:
        sha = run("rev-parse", "--short", "HEAD").strip()
        dirty = run("status", "--porcelain", "--", *CODE_PATHS).strip()
    except Exception:
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
