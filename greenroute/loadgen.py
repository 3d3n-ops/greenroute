"""Open-loop load generator for an OpenAI-compatible vLLM server.

Requests arrive on a Poisson schedule at a target rate regardless of how fast the
server answers (open loop), the way real traffic does. Each request streams so we
can time the first token and the decode rate.
"""

from __future__ import annotations

import asyncio
import random
import re
import time

# vLLM Prometheus gauges worth tracking. Names changed across versions, so both are listed.
VLLM_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
)
_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")


def poisson_schedule(n: int, rate: float, seed: int = 0) -> list[float]:
    """Send times in seconds from start. rate <= 0 means a burst: everything at t=0."""
    if rate <= 0:
        return [0.0] * n
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out


def percentiles(xs: list[float], ps=(50, 90, 99)) -> dict:
    """Linear-interpolated percentiles, keyed p50/p90/p99. Empty input gives {}."""
    if not xs:
        return {}
    s = sorted(xs)
    out = {}
    for p in ps:
        k = (len(s) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        out[f"p{p}"] = s[lo] + (s[hi] - s[lo]) * (k - lo)
    return out


def parse_vllm_metrics(text: str) -> dict:
    """Sum each tracked gauge across label sets from a Prometheus text dump."""
    out: dict = {}
    for line in text.splitlines():
        m = _METRIC_LINE.match(line.strip())
        if not m or m.group(1) not in VLLM_GAUGES:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


async def _one_request(client, model, idx, messages, send_at, t0, max_tokens, thinking: bool = False) -> dict:
    await asyncio.sleep(max(0.0, t0 + send_at - time.time()))
    t_send = time.time()
    t_first = None
    parts: list[str] = []
    completion_tokens = None
    error = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if t_first is None:
                    t_first = time.time()
                parts.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                completion_tokens = chunk.usage.completion_tokens
    except Exception as e:  # recorded, not raised: one failure shouldn't end the level
        error = repr(e)
    t_end = time.time()

    ttft = (t_first - t_send) if t_first is not None else None
    tpot = None
    if t_first is not None and completion_tokens and completion_tokens > 1:
        tpot = (t_end - t_first) / (completion_tokens - 1)
    return {
        "index": idx,
        "scheduled_s": send_at,
        "sent_s": t_send - t0,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": t_end - t_send,
        "output_tokens": completion_tokens or 0,
        "text": "".join(parts),
        "error": error,
    }


async def _scrape_metrics(base_url: str, interval: float, stop: asyncio.Event, t0: float, out: list) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=5.0) as http:
        while not stop.is_set():
            try:
                r = await http.get(f"{base_url}/metrics")
                out.append({"t": time.time() - t0, **parse_vllm_metrics(r.text)})
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


def _worker_init() -> None:
    """Import the heavy client stack before the clock starts."""
    import httpx  # noqa: F401
    import openai  # noqa: F401


def _worker(base_url: str, model: str, items: list, t0: float, max_tokens: int, thinking: bool = False) -> dict:
    """One process: its own event loop and HTTP client for a slice of the schedule.

    Also reports the share of one core this process used from the first scheduled send
    to the end. Near 1.0 means the sender, not the server, was the limit.
    """
    cpu0 = time.process_time()
    records = asyncio.run(_worker_async(base_url, model, items, t0, max_tokens, thinking))
    busy_s = max(time.time() - t0, 1e-9)
    return {"records": records, "cpu_frac": (time.process_time() - cpu0) / busy_s}


async def _worker_async(base_url, model, items, t0, max_tokens, thinking=False) -> list[dict]:
    import httpx
    from openai import AsyncOpenAI

    http = httpx.AsyncClient(
        timeout=1800.0, limits=httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    )
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="unused", http_client=http, max_retries=0)
    try:
        return list(await asyncio.gather(
            *(_one_request(client, model, idx, m, send_at, t0, max_tokens, thinking) for idx, m, send_at in items)
        ))
    finally:
        await client.close()


def make_pool(workers: int):
    """Worker processes for the load generator, started and warmed before measurement."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init)
    list(pool.map(abs, range(workers * 2)))  # force every worker to start and import
    return pool


async def run_level(
    base_url: str,
    model: str,
    message_lists: list[list[dict]],
    rate: float,
    pool,
    workers: int,
    seed: int = 0,
    max_tokens: int = 1024,
    metrics_interval: float = 0.5,
    lead_s: float = 0.2,
    thinking: bool = False,
) -> dict:
    """Send every message list at `rate` req/s (Poisson; <= 0 = burst) across `workers` processes.

    Requests are dealt round-robin to workers; all share one wall-clock schedule, so the
    arrival process is the same as a single sender's. Returns per-request records.
    """
    schedule = poisson_schedule(len(message_lists), rate, seed)
    items = [(i, m, s) for i, (m, s) in enumerate(zip(message_lists, schedule))]
    slices = [items[w::workers] for w in range(workers)]

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    metric_samples: list[dict] = []
    t0 = time.time() + lead_s
    scraper = asyncio.create_task(_scrape_metrics(base_url, metrics_interval, stop, t0, metric_samples))
    futures = [
        loop.run_in_executor(pool, _worker, base_url, model, sl, t0, max_tokens, thinking) for sl in slices if sl
    ]
    results = await asyncio.gather(*futures)
    wall = time.time() - t0
    stop.set()
    await scraper
    records = sorted((r for part in results for r in part["records"]), key=lambda r: r["index"])
    return {
        "records": records,
        "metrics": metric_samples,
        "wall_s": wall,
        "t0": t0,  # wall-clock start; record times are relative to it
        "client_cpu_frac_max": max((part["cpu_frac"] for part in results), default=None),
    }


def send_lag(records: list[dict]) -> list[float]:
    """How late each request left the sender versus its schedule. Growth means a starved client."""
    return [r["sent_s"] - r["scheduled_s"] for r in records]


def summarize_metrics(samples: list[dict]) -> dict:
    """Mean and max of each vLLM gauge over the level."""
    out = {}
    for g in VLLM_GAUGES:
        vals = [s[g] for s in samples if g in s]
        if vals:
            short = g.split(":", 1)[1]
            out[f"{short}_mean"] = sum(vals) / len(vals)
            out[f"{short}_max"] = max(vals)
    return out
