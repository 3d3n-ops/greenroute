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


def _fake_vllm(monkeypatch):
    import sys
    import types

    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = types.SimpleNamespace(DELTA="delta")
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)


class _Params:
    def __init__(self, seed):
        self.seed, self.output_kind = seed, None

    def clone(self):
        return _Params(self.seed)


class _Engine:
    """Finishes one request per step; each emits a token equal to its seed."""

    def __init__(self):
        self.pending = []

    def add_request(self, rid, prompt, params):
        assert params.output_kind == "delta"
        self.pending.append((rid, params.seed))

    def has_unfinished_requests(self):
        return bool(self.pending)

    def get_num_unfinished_requests(self):
        return len(self.pending)

    def step(self):
        from types import SimpleNamespace as NS

        rid, seed = self.pending.pop(0)
        o = NS(text=str(seed), token_ids=[seed], logprobs=None, finish_reason="stop")
        return [NS(request_id=rid, outputs=[o], finished=True)]


class _Meter:
    def read_mj(self):
        return 0.0


def test_run_engine_takes_one_params_per_prompt(monkeypatch):
    from types import SimpleNamespace as NS

    _fake_vllm(monkeypatch)
    llm = NS(llm_engine=_Engine())
    outs, timeline = steady.run_engine(llm, [[1], [2], [3]], [_Params(7), _Params(8), _Params(9)], _Meter())
    assert [o.outputs[0].token_ids for o in outs] == [[7], [8], [9]]
    assert [row[2] for row in timeline] == [3, 2, 1, 0] and timeline[-1][3] == 3

    with pytest.raises(ValueError):
        steady.run_engine(NS(llm_engine=_Engine()), [[1], [2]], [_Params(1)], _Meter())


class _LongTailEngine(_Engine):
    """One request that decodes `steps` tokens before finishing, as in a drain-out tail."""

    def __init__(self, steps):
        super().__init__()
        self.left = steps

    def step(self):
        from types import SimpleNamespace as NS

        rid, seed = self.pending[0]
        self.left -= 1
        done = self.left == 0
        if done:
            self.pending.pop(0)
        o = NS(text="x", token_ids=[seed], logprobs=None, finish_reason="stop" if done else None)
        return [NS(request_id=rid, outputs=[o], finished=done)]


class _CountingMeter(_Meter):
    def __init__(self):
        self.reads = 0

    def read_mj(self):
        self.reads += 1
        return 0.0


def test_run_engine_does_not_read_the_meter_every_step_of_a_long_tail(monkeypatch):
    from types import SimpleNamespace as NS

    _fake_vllm(monkeypatch)
    meter = _CountingMeter()
    outs, timeline = steady.run_engine(NS(llm_engine=_LongTailEngine(500)), [[1]], _Params(4), meter)
    assert len(outs[0].outputs[0].token_ids) == 500
    # 500 steps run in well under TIMELINE_MAX_GAP_S here, so only the start and the finish are read.
    assert meter.reads < 10 and timeline[-1][2] == 0 and timeline[-1][3] == 500
