import time
from types import SimpleNamespace

import pytest

from greenroute.energy import EnergyMeter, EnergyReading, TelemetrySampler
from greenroute.report import summarize


class FakeNVML:
    """Stands in for pynvml; each counter read advances by `step_mj` per GPU."""

    def __init__(self, gpus=2, step_mj=5000):
        self.gpus = gpus
        self.step_mj = step_mj
        self.counters = [0] * gpus

    def nvmlInit(self):
        pass

    def nvmlDeviceGetCount(self):
        return self.gpus

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self.counters[h] += self.step_mj
        return self.counters[h]

    def nvmlDeviceGetName(self, h):
        return b"FAKE-GPU"

    def nvmlSystemGetDriverVersion(self):
        return "1.0"

    # Telemetry reads
    NVML_CLOCK_SM = 1
    NVML_TEMPERATURE_GPU = 0

    def nvmlDeviceGetPowerUsage(self, h):
        return 250_000  # mW

    def nvmlDeviceGetClockInfo(self, h, kind):
        assert kind == self.NVML_CLOCK_SM
        return 1980

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=87)

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(used=72e9)

    def nvmlDeviceGetTemperature(self, h, kind):
        return 55


def test_span_sums_gpus_in_joules():
    meter = EnergyMeter(nvml=FakeNVML(gpus=2, step_mj=5000))
    with meter.span() as s:
        pass
    assert s.reading.per_gpu_joules == [5.0, 5.0]
    assert s.reading.joules == 10.0


def test_selected_gpu_only():
    meter = EnergyMeter(gpu_indices=[1], nvml=FakeNVML(gpus=2))
    with meter.span() as s:
        pass
    assert len(s.reading.per_gpu_joules) == 1


def test_device_info_decodes_bytes():
    meter = EnergyMeter(gpu_indices=[0], nvml=FakeNVML())
    assert meter.device_info() == [{"name": "FAKE-GPU", "driver": "1.0"}]


def test_active_joules_subtracts_idle_floor():
    r = EnergyReading(joules=1000.0, seconds=10.0)
    assert r.avg_watts == 100.0
    assert r.active_joules(idle_watts=60.0) == 400.0
    assert r.active_joules(idle_watts=200.0) == 0.0


def test_summarize_headline_numbers():
    r = EnergyReading(joules=1000.0, seconds=10.0)
    s = summarize(r, idle_watts=60.0, n_tasks=50, n_correct=40, output_tokens=10000)
    assert s["accuracy"] == 0.8
    assert s["j_per_task_total"] == 20.0
    assert s["j_per_solved_task_total"] == 25.0
    assert s["j_per_solved_task_active"] == 10.0
    assert s["j_per_output_token_total"] == pytest.approx(0.1)
    assert s["tasks_per_hour"] == 18000.0


def test_summarize_zero_correct_is_none_not_crash():
    r = EnergyReading(joules=100.0, seconds=1.0)
    s = summarize(r, idle_watts=0.0, n_tasks=5, n_correct=0, output_tokens=0)
    assert s["j_per_solved_task_total"] is None
    assert s["j_per_output_token_total"] is None


def test_telemetry_sampler_collects_and_summarizes():
    meter = EnergyMeter(nvml=FakeNVML(gpus=1))
    with TelemetrySampler(meter, interval=0.01) as tel:
        time.sleep(0.08)
    assert len(tel.samples) >= 3
    s0 = tel.samples[0]
    assert s0["power_w"] == 250.0 and s0["sm_clock_mhz"] == 1980 and s0["mem_used_gb"] == 72.0
    summary = tel.summary()
    assert summary["power_w_mean"] == 250.0
    assert summary["util_pct_max"] == 87
    assert summary["n_samples"] == len(tel.samples)


def test_telemetry_summary_empty_is_safe():
    tel = TelemetrySampler(EnergyMeter(nvml=FakeNVML(gpus=1)))
    assert tel.summary() == {"n_samples": 0}
