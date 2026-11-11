"""Charts for a Sprint 2 run directory (latency vs energy under server traffic).

    python analysis/plot_sprint2.py results/sprint2/20261111-123849
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COLORS = {"Qwen3-8B": BLUE, "Qwen3-1.7B": ORANGE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def load(run_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))}


def steady_levels(result: dict) -> list[dict]:
    """Poisson levels only; the burst is a different experiment (everything at t=0)."""
    return [lv for lv in result["levels"] if lv["rate"] > 0]


def energy_vs_rate(runs: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = None
    for name in ("Qwen3-8B", "Qwen3-1.7B"):
        lvs = steady_levels(runs[name])
        xs = [lv["rate"] for lv in lvs]
        ys = [lv["energy"]["j_per_task_total"] for lv in lvs]
        rates = xs
        ax.plot(xs, ys, color=COLORS[name], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
        ax.annotate(f"{name.replace('Qwen3-', '')}  {ys[0]:.0f} J", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(10, 2), color=INK, fontsize=10)
        ax.annotate(f"{ys[-1]:.1f} J", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 4), color=INK, fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(rates))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{r:g}" for r in rates])
    ax.set_yticks([5, 10, 20, 50, 100, 200, 500])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.8, 90)
    ax.set_xlabel("Offered traffic (requests per second, Poisson arrivals)")
    ax.set_ylabel("Joules per GSM8K request (total GPU energy, log)")
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("A quiet GPU is expensive: energy per request falls 30× with traffic",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    ax.set_title("vLLM server on one H100 per model, streaming GSM8K requests, thinking off\n"
                 "Same model, same hardware: only the traffic level changes",
                 loc="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def tradeoff_8b(runs: dict[str, dict], out: Path) -> None:
    lvs = steady_levels(runs["Qwen3-8B"])
    xs = [lv["energy"]["j_per_task_total"] for lv in lvs]
    p50 = [lv["latency"]["e2e_s"]["p50"] for lv in lvs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, p50, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
    for lv, x, y in zip(lvs, xs, p50):
        last = lv is lvs[-1]
        ax.annotate(f"{lv['rate']:g} rps", (x, y), textcoords="offset points",
                    xytext=(-10, 0) if last else (0, -18), ha="right" if last else "center",
                    va="center" if last else "baseline", color=INK, fontsize=9.5)
    detail = "\n".join(
        f"{lv['rate']:>3g} rps   first token {lv['latency']['ttft_s']['p50'] * 1000:3.0f} ms · "
        f"{lv['latency']['tpot_s']['p50'] * 1000:4.1f} ms per token"
        for lv in lvs if lv["rate"] in (1, 16, 64)
    )
    ax.text(0.02, 0.97, "Median per request\n" + detail, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, color=INK_2, family="monospace")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(["10", "20", "50", "100", "200", "500"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(600, 10)  # reversed: left to right = more efficient
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("Joules per request (total GPU energy, log; lower is better →)")
    ax.set_ylabel("Median end-to-end time per request (s)")
    fig.suptitle("Qwen3-8B: 30× less energy per request costs 1.8× median latency",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    ax.set_title("One H100, vLLM server, ~256 output tokens per request. Each point is one traffic level\n"
                 "Latency stays flat to 16 rps; the knee is between 32 and 64 rps",
                 loc="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    runs = load(run_dir)
    energy_vs_rate(runs, run_dir / "energy_vs_traffic.png")
    tradeoff_8b(runs, run_dir / "latency_energy_tradeoff_8b.png")
    print(f"wrote {run_dir / 'energy_vs_traffic.png'} and {run_dir / 'latency_energy_tradeoff_8b.png'}")
