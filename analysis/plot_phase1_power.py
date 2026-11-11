"""Chart for a phase1_power run: energy per task against traffic, fast vs thinking, per model size.

    python analysis/plot_phase1_power.py results/phase1_power/20261111-172237
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODES = (("fast", False, BLUE), ("thinking", True, ORANGE))
SIZES = ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B")
OVERLOADED_TTFT_P99_S = 2.0  # first-token p99 above this: the server is queueing, not keeping up

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def load(run_dir: Path) -> dict[tuple[str, bool], list[dict]]:
    out = defaultdict(list)
    with open(run_dir / "summary.csv") as f:
        for r in csv.DictReader(f):
            key = (r["model"].split("/")[-1], r["thinking"] == "True")
            out[key].append({k: (v if k in ("label", "model", "thinking") else float(v)) for k, v in r.items()})
    for rows in out.values():
        rows.sort(key=lambda r: r["rate"])
    return out


def energy_by_mode(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
    rates = sorted({r["rate"] for rows in data.values() for r in rows})
    for ax, size in zip(axes, SIZES):
        by_rate = {}
        for _, thinking, color in MODES:
            rows = data[(size, thinking)]
            xs = [r["rate"] for r in rows]
            ys = [r["steady_j_per_task"] for r in rows]
            ax.plot(xs, ys, color=color, lw=2, zorder=2)
            for r, x, y in zip(rows, xs, ys):
                over = r["ttft_p99"] > OVERLOADED_TTFT_P99_S
                ax.plot([x], [y], marker="o", ms=7, color=color, mec=color if over else SURFACE,
                        mfc=SURFACE if over else color, mew=2 if over else 1.5, zorder=3)
            by_rate[thinking] = dict(zip(xs, ys))
        for rate in (1, 8):
            fast, think = by_rate[False].get(rate), by_rate[True].get(rate)
            if fast and think:
                ax.annotate(f"{think / fast:.1f}×", (rate, think), textcoords="offset points", xytext=(0, 9),
                            ha="center", color=INK, fontsize=10)
        ax.set_title(size, loc="left", fontsize=11, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(rates))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{r:g}" for r in rates], fontsize=9)
        ax.set_xlim(0.2, 80)
        ax.set_xlabel("Offered requests per second")
    axes[0].set_yticks([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    axes[0].set_yticklabels(["5", "10", "20", "50", "100", "200", "500", "1k", "2k"])
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("Joules per task (log)")
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=7, mec=SURFACE, label=n) for n, _, c in MODES]
    handles.append(Line2D([], [], color=INK_2, lw=0, marker="o", ms=7, mfc=SURFACE, mec=INK_2, mew=2,
                          label=f"overloaded (first-token p99 > {OVERLOADED_TTFT_P99_S:g} s)"))
    fig.legend(handles=handles, loc="upper left", frameon=False, ncol=3, fontsize=10, bbox_to_anchor=(0.0, 0.8))
    fig.suptitle("At server traffic, thinking costs 1.3–2.1× a direct answer, not 8–12×",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.885, "vLLM server on one H100 per configuration, Poisson GSM8K traffic. "
             "Labels: thinking ÷ fast at 1 and 8 req/s.\nThe GPU's fixed power is paid either way, "
             "so longer reasoning mostly fills capacity that would sit idle.",
             ha="left", va="top", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.73))
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    energy_by_mode(load(run_dir), run_dir / "energy_fast_vs_thinking.png")
    print(f"wrote {run_dir / 'energy_fast_vs_thinking.png'}")
