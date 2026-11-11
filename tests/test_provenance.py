import subprocess

from greenroute.provenance import git_sha


def _git(cwd, *args):
    subprocess.check_call(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                          cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_git_sha_marks_uncommitted_code(tmp_path):
    assert git_sha(cwd=str(tmp_path)) == "unknown"  # not a repo

    (tmp_path / "greenroute").mkdir()
    (tmp_path / "greenroute" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    clean = git_sha(cwd=str(tmp_path))
    assert clean != "unknown" and not clean.endswith("-dirty")

    (tmp_path / "results.json").write_text("{}")       # results do not make the code dirty
    assert git_sha(cwd=str(tmp_path)) == clean

    (tmp_path / "greenroute" / "b.py").write_text("y = 2\n")  # new, uncommitted code does
    assert git_sha(cwd=str(tmp_path)) == clean + "-dirty"
