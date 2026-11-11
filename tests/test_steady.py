import pytest

from greenroute import steady
from greenroute.steady import downsample, steady_window

# (seconds, joules, unfinished, tokens): 6 requests, cap 2.
# Backlogged while unfinished >= 2 (rows 0-4); 2 requests have finished by row 2 (turnover);
# then a slow one-request tail.
TIMELINE = [
    (0.0, 0.0, 6, 0),
    (1.0, 500.0, 6, 100),
    (2.0, 1000.0, 4, 200),
    (3.0, 1600.0, 3, 300),
    (4.0, 2200.0, 2, 400),
    (5.0, 2500.0, 1, 420),
    (7.0, 3100.0, 0, 460),
]


def test_backlogged_turnover_and_tail(monkeypatch):
    monkeypatch.setattr(steady, "MIN_TURNOVER_TOKEN_SHARE", 0.1)
    s = steady_window(TIMELINE, threshold=2)
    assert s["end_index"] == 4 and s["turnover_start_index"] == 2
    b, t, tail = s["backlogged"], s["turnover"], s["tail"]
    assert (b["seconds"], b["energy_j"], b["output_tokens"]) == (4.0, 2200.0, 400)
    assert b["j_per_output_token"] == pytest.approx(5.5)
    # Turnover window (rows 2-4): 2 s, 1200 J, 200 tokens -> 6 J/token, dearer than the young start.
    assert (t["seconds"], t["energy_j"], t["output_tokens"]) == (2.0, 1200.0, 200)
    assert s["basis"] == "turnover" and s["j_per_output_token"] == pytest.approx(6.0)
    # Tail: 3 s, 900 J, 60 tokens -> 15 J/token.
    assert (tail["seconds"], tail["energy_j"], tail["output_tokens"]) == (3.0, 900.0, 60)
    assert s["share_of_tokens"] == pytest.approx(400 / 460)


def test_short_turnover_window_falls_back_to_backlogged(monkeypatch):
    monkeypatch.setattr(steady, "MIN_TURNOVER_TOKEN_SHARE", 0.5)  # 200/460 tokens is not enough
    s = steady_window(TIMELINE, threshold=2)
    assert s["basis"] == "backlogged" and s["j_per_output_token"] == pytest.approx(5.5)


def test_no_window_when_run_never_fills_the_cap():
    assert steady_window(TIMELINE, threshold=7) is None  # fewer problems than slots
    assert steady_window(TIMELINE[:1], threshold=1) is None


def test_downsample_keeps_ends_and_requested_rows():
    tl = [(float(i), float(i), 100 - i, i) for i in range(100)]
    d = downsample(tl, max_rows=10, keep=(37,))
    assert d[0] == tl[0] and d[-1] == tl[-1]
    assert (37.0, 37.0, 63, 37) in d
    assert len(d) <= 13
    assert downsample(tl[:5], max_rows=10) == tl[:5]
