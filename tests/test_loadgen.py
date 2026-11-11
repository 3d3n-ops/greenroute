import pytest

from greenroute.loadgen import parse_vllm_metrics, percentiles, poisson_schedule, summarize_metrics
from greenroute.serve_bench import requests_for_rate


def test_poisson_schedule_rate_and_order():
    s = poisson_schedule(5000, rate=10.0, seed=1)
    assert s[0] == 0.0
    assert all(b >= a for a, b in zip(s, s[1:]))
    assert s[-1] / (len(s) - 1) == pytest.approx(0.1, rel=0.05)  # mean gap ~ 1/rate


def test_poisson_schedule_is_reproducible_and_burst():
    assert poisson_schedule(10, 2.0, seed=3) == poisson_schedule(10, 2.0, seed=3)
    assert poisson_schedule(4, 0) == [0.0, 0.0, 0.0, 0.0]


def test_percentiles_interpolate():
    p = percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert p["p50"] == pytest.approx(5.5)
    assert p["p90"] == pytest.approx(9.1)
    assert percentiles([]) == {}
    assert percentiles([3.0]) == {"p50": 3.0, "p90": 3.0, "p99": 3.0}


def test_parse_vllm_metrics_sums_labels_and_skips_others():
    text = """
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-1.7B"} 12.0
vllm:num_requests_running{engine="1",model_name="Qwen/Qwen3-1.7B"} 3.0
vllm:num_requests_waiting{model_name="Qwen/Qwen3-1.7B"} 0.0
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen3-1.7B"} 0.25
vllm:prompt_tokens_total{model_name="Qwen/Qwen3-1.7B"} 9999.0
"""
    m = parse_vllm_metrics(text)
    assert m == {
        "vllm:num_requests_running": 15.0,
        "vllm:num_requests_waiting": 0.0,
        "vllm:gpu_cache_usage_perc": 0.25,
    }
    s = summarize_metrics([m, {**m, "vllm:num_requests_running": 5.0}])
    assert s["num_requests_running_mean"] == 10.0
    assert s["num_requests_running_max"] == 15.0


def test_requests_for_rate_clamps():
    assert requests_for_rate(1, 90, 100, 600) == 100
    assert requests_for_rate(4, 90, 100, 600) == 360
    assert requests_for_rate(64, 90, 100, 600) == 600
    assert requests_for_rate(0, 90, 100, 600) == 600


def _chunk(content=None, usage=None):
    from types import SimpleNamespace as NS
    choices = [NS(delta=NS(content=content))] if usage is None else []
    return NS(choices=choices, usage=usage)


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        import asyncio
        if not self._chunks:
            raise StopAsyncIteration
        await asyncio.sleep(0.001)
        return self._chunks.pop(0)


class _FakeClient:
    def __init__(self, chunks=None, fail=False):
        from types import SimpleNamespace as NS
        self.kwargs = None
        outer = self

        async def create(**kwargs):
            outer.kwargs = kwargs
            if fail:
                raise RuntimeError("boom")
            return _FakeStream(chunks)

        self.chat = NS(completions=NS(create=create))


def test_one_request_times_stream_and_counts_tokens():
    import asyncio
    import time
    from types import SimpleNamespace as NS

    from greenroute.loadgen import _one_request

    chunks = [_chunk(""), _chunk("6 + 2 = 8"), _chunk("\nAnswer: 8"), _chunk(usage=NS(completion_tokens=5))]
    client = _FakeClient(chunks)
    r = asyncio.run(_one_request(client, "m", 0, [{"role": "user", "content": "q"}], 0.0, time.time(), 64))
    assert r["error"] is None
    assert r["text"] == "6 + 2 = 8\nAnswer: 8"
    assert r["output_tokens"] == 5
    assert 0 < r["ttft_s"] <= r["e2e_s"]
    assert r["tpot_s"] > 0
    assert client.kwargs["stream"] is True
    assert client.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_one_request_records_errors_instead_of_raising():
    import asyncio
    import time

    from greenroute.loadgen import _one_request

    r = asyncio.run(_one_request(_FakeClient(fail=True), "m", 3, [], 0.0, time.time(), 64))
    assert r["error"] and "boom" in r["error"]
    assert r["ttft_s"] is None and r["output_tokens"] == 0


def test_worker_pool_round_trip(monkeypatch):
    """The multi-process path: requests dealt across workers, records come back in index order."""
    import asyncio

    from greenroute import loadgen

    pool = loadgen.make_pool(2)
    try:
        # Unreachable server: every request errors fast, which exercises the full path without a GPU.
        msgs = [[{"role": "user", "content": str(i)}] for i in range(6)]
        out = asyncio.run(loadgen.run_level("http://127.0.0.1:9", "m", msgs, rate=0, pool=pool, workers=2,
                                            metrics_interval=0.05))
    finally:
        pool.shutdown()
    assert [r["index"] for r in out["records"]] == list(range(6))
    assert all(r["error"] for r in out["records"])
    assert out["wall_s"] > 0
    assert out["client_cpu_frac_max"] >= 0
    lag = loadgen.send_lag(out["records"])
    assert len(lag) == 6 and all(x > -0.05 for x in lag)  # burst: sent at, not before, the shared start


def test_common_accuracy_uses_only_the_shared_problems():
    from greenroute.serve_bench import common_accuracy

    recs = [{"index": i, "correct": i < 3} for i in range(10)]
    assert common_accuracy(recs, 4) == 0.75   # 3 of the first 4
    assert common_accuracy(recs, 10) == 0.3   # same model, bigger subset, different number
    assert common_accuracy([], 4) is None
