"""Turn a measured run into the headline numbers."""

from __future__ import annotations

from greenroute.energy import EnergyReading


def summarize(
    reading: EnergyReading,
    idle_watts: float,
    n_tasks: int,
    n_correct: int,
    output_tokens: int,
) -> dict:
    """Energy per task, per solved task (J/task@Q for a single model) and per token.

    `total` counts everything the GPU drew during the run, idle floor included.
    `active` subtracts the idle baseline over the same span. Both are reported;
    neither is hidden.
    """
    total = reading.joules
    active = reading.active_joules(idle_watts)

    def per(x: float, n: int) -> float | None:
        return x / n if n else None

    return {
        "n_tasks": n_tasks,
        "n_correct": n_correct,
        "accuracy": per(n_correct, n_tasks),
        "seconds": reading.seconds,
        "avg_watts": reading.avg_watts,
        "idle_watts": idle_watts,
        "energy_j_total": total,
        "energy_j_active": active,
        "j_per_task_total": per(total, n_tasks),
        "j_per_solved_task_total": per(total, n_correct),
        "j_per_solved_task_active": per(active, n_correct),
        "output_tokens": output_tokens,
        "j_per_output_token_total": per(total, output_tokens),
        "tasks_per_hour": n_tasks / reading.seconds * 3600 if reading.seconds else None,
    }
