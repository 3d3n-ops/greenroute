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
    assert meter.device_info() == [{"name": "FAKE-GPU", "driver": "1.0", "metered_by": "explicit"}]


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


class FakeNVMLWithUUID(FakeNVML):
    def nvmlDeviceGetHandleByUUID(self, uuid):
        return int(uuid.rsplit("-", 1)[1])

    def nvmlDeviceGetUUID(self, h):
        return f"GPU-fake-{h}"

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        return 700_000

    def nvmlDeviceGetPowerManagementDefaultLimit(self, h):
        return 700_000


def test_meter_reads_only_the_gpus_this_process_uses(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    m = EnergyMeter(nvml=FakeNVMLWithUUID(gpus=4))
    assert m._handles == [2] and m.selection == "CUDA_VISIBLE_DEVICES=2"
    info = m.device_info()
    assert info[0]["uuid"] == "GPU-fake-2" and info[0]["power_limit_w"] == 700.0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fake-1,GPU-fake-3")
    assert EnergyMeter(nvml=FakeNVMLWithUUID(gpus=4))._handles == [1, 3]


def test_meter_falls_back_to_every_gpu_when_it_cannot_map(monkeypatch):
    # Indices without PCI bus order may not match NVML's numbering: meter everything.
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    m = EnergyMeter(nvml=FakeNVMLWithUUID(gpus=4))
    assert m._handles == [0, 1, 2, 3] and m.selection == "all"
    # One GPU in the container (as on Modal): index 0 is the only one either way.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert EnergyMeter(nvml=FakeNVMLWithUUID(gpus=1))._handles == [0]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert EnergyMeter(nvml=FakeNVMLWithUUID(gpus=2))._handles == [0, 1]
    # Older fakes / drivers without UUID or power-limit calls still describe the device.
    assert EnergyMeter(nvml=FakeNVML(gpus=1)).device_info()[0]["name"] == "FAKE-GPU"
