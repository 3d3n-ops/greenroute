"""Charts for a Sprint 1 run directory.

    python analysis/plot_sprint1.py results/sprint1/20261111-113649
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def load(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]


def batch_sweep(runs: list[dict], out: Path) -> None:
    small = [r for r in runs if r["config"]["model"].endswith("1.7B")]
    by_seqs: dict[int, list[dict]] = {}
    for r in small:
        by_seqs.setdefault(r["config"]["max_num_seqs"], []).append(r["summary"])
    xs = sorted(by_seqs)

    def stat(key):
        vals = [[s[key] for s in by_seqs[x]] for x in xs]
        return [statistics.mean(v) for v in vals], [min(v) for v in vals], [max(v) for v in vals]

    tot, tot_lo, tot_hi = stat("j_per_solved_task_total")
    act, _, _ = stat("j_per_solved_task_active")
    acc = [statistics.mean(s["accuracy"] for s in by_seqs[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, tot, yerr=[[t - lo for t, lo in zip(tot, tot_lo)], [hi - t for t, hi in zip(tot, tot_hi)]],
                color=BLUE, lw=2, marker="o", ms=8, capsize=4, mec=SURFACE, mew=2, label="Total GPU energy")
    ax.plot(xs, act, color=ORANGE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="Above idle power")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Concurrent requests (vLLM max_num_seqs)")
    ax.set_ylabel("Joules per solved GSM8K problem (log)")

    for x, t in [(xs[0], tot[0]), (xs[-1], tot[-1])]:
        ax.annotate(f"{t:.1f} J", (x, t), textcoords="offset points", xytext=(10, 6), color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ratio = tot[0] / tot[-1]
    fig.suptitle(f"Batching cuts energy per solved task {ratio:.0f}× at the same accuracy",
                 x=0.06, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(f"Qwen3-1.7B on one H100, all 1,319 GSM8K test problems · accuracy {min(acc):.1%}–{max(acc):.1%} at every batch size"
                 f"\nGPU energy via NVML · batch 32 shows mean and range of 3 runs",
                 loc="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def model_sizes(runs: list[dict], out: Path) -> None:
    pts = {}
    for r in runs:
        c, s = r["config"], r["summary"]
        if c["max_num_seqs"] == 32:
            pts.setdefault(c["model"].split("/")[-1], []).append(s)
    best = max((r for r in runs if r["config"]["model"].endswith("1.7B")), key=lambda r: r["config"]["max_num_seqs"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"):
        ss = pts[name]
        x = statistics.mean(s["j_per_solved_task_total"] for s in ss)
        y = statistics.mean(s["accuracy"] for s in ss) * 100
        ax.plot(x, y, "o", color=BLUE, ms=10, mec=SURFACE, mew=2)
        ax.annotate(f"{name.replace('Qwen3-', '')} · batch 32", (x, y), textcoords="offset points", xytext=(10, -4), color=INK)
    bx, by = best["summary"]["j_per_solved_task_total"], best["summary"]["accuracy"] * 100
    ax.plot(bx, by, "o", mfc=SURFACE, mec=BLUE, ms=10, mew=2)
    ax.annotate(f"1.7B · batch {best['config']['max_num_seqs']}", (bx, by), textcoords="offset points", xytext=(-8, 12), color=INK)

    ax.set_xlim(0, 42)
    ax.set_ylim(78, 96)
    ax.set_xlabel("Joules per solved GSM8K problem (total GPU energy)")
    ax.set_ylabel("GSM8K accuracy (%)")
    fig.suptitle("Model size buys accuracy; batching cuts energy without losing it",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    ax.set_title("Qwen3 family on one H100, all 1,319 GSM8K test problems, thinking off\n"
                 "Filled: batch 32 (1.7B is mean of 3 runs) · hollow: same 1.7B model at batch 256",
                 loc="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    runs = load(run_dir)
    batch_sweep(runs, run_dir / "batch_sweep.png")
    model_sizes(runs, run_dir / "model_sizes.png")
    print(f"wrote {run_dir / 'batch_sweep.png'} and {run_dir / 'model_sizes.png'}")
