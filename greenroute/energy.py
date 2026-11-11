"""GPU energy measurement via NVML's cumulative energy counter.

`nvmlDeviceGetTotalEnergyConsumption` returns millijoules consumed since the
driver was loaded (Volta and newer). Reading it before and after a workload is
more accurate than integrating sampled power draw, which the GPU averages
internally over a window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EnergyReading:
    """Energy used by a measured span, summed over the metered GPUs."""

    joules: float
    seconds: float
    per_gpu_joules: list[float] = field(default_factory=list)

    @property
    def avg_watts(self) -> float:
        return self.joules / self.seconds if self.seconds > 0 else 0.0

    def active_joules(self, idle_watts: float) -> float:
        """Energy above the idle baseline over the same span."""
        return max(self.joules - idle_watts * self.seconds, 0.0)


class EnergyMeter:
    """Measure GPU energy for a span of code.

    Usage:
        meter = EnergyMeter()
        with meter.span() as span:
            run_workload()
        print(span.reading.joules)
    """

    def __init__(self, gpu_indices: list[int] | None = None, nvml=None):
        if nvml is None:
            import pynvml as nvml  # provided by the nvidia-ml-py package
        self._nvml = nvml
        self._nvml.nvmlInit()
        count = self._nvml.nvmlDeviceGetCount()
        indices = gpu_indices if gpu_indices is not None else list(range(count))
        self._handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in indices]

    def _counters_mj(self) -> list[int]:
        return [self._nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self._handles]

    def span(self) -> "_Span":
        return _Span(self)

    def idle_watts(self, seconds: float = 10.0) -> float:
        """Average total power of the metered GPUs while nothing runs."""
        with self.span() as s:
            time.sleep(seconds)
        return s.reading.avg_watts

    def device_info(self) -> list[dict]:
        info = []
        for h in self._handles:
            name = self._nvml.nvmlDeviceGetName(h)
            info.append({"name": name.decode() if isinstance(name, bytes) else name})
        driver = self._nvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        for d in info:
            d["driver"] = driver
        return info


class _Span:
    def __init__(self, meter: EnergyMeter):
        self._meter = meter
        self.reading: EnergyReading | None = None

    def __enter__(self) -> "_Span":
        self._t0 = time.perf_counter()
        self._e0 = self._meter._counters_mj()
        return self

    def __exit__(self, *exc) -> None:
        e1 = self._meter._counters_mj()
        seconds = time.perf_counter() - self._t0
        per_gpu = [(b - a) / 1000.0 for a, b in zip(self._e0, e1)]
        self.reading = EnergyReading(joules=sum(per_gpu), seconds=seconds, per_gpu_joules=per_gpu)
