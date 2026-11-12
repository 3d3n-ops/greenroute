import os

from greenroute.env import load_env, parse


def test_parse_forms():
    text = """
# comment
A=1
export B = two
C="quoted # not a comment"
D='single'
E=plain # trailing comment
F=            # only a comment
not a line
"""
    assert parse(text) == {"A": "1", "B": "two", "C": "quoted # not a comment", "D": "single", "E": "plain", "F": ""}


def test_load_env_respects_shell_and_skips_empty(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("GR_TEST_NEW=x\nGR_TEST_SHELL=file\nGR_TEST_EMPTY=\n")
    monkeypatch.setenv("GR_TEST_SHELL", "shell")
    monkeypatch.delenv("GR_TEST_NEW", raising=False)
    monkeypatch.delenv("GR_TEST_EMPTY", raising=False)
    assert load_env(f) == ["GR_TEST_NEW"]
    assert os.environ["GR_TEST_NEW"] == "x"
    assert os.environ["GR_TEST_SHELL"] == "shell"
    assert "GR_TEST_EMPTY" not in os.environ
    monkeypatch.delenv("GR_TEST_NEW")


def test_missing_file_is_noop(tmp_path):
    assert load_env(tmp_path / "nope") == []
