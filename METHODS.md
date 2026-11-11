# Methods (draft)

This is a living one-pager. Critique is welcome — open an issue.

## Metric

**J/task@Q** — GPU joules to complete one task, at a fixed quality floor, with retries and escalations included.

- A task counts as complete only if it passes the eval (correct GSM8K answer, passing HumanEval tests).
- If a small model fails and the request escalates to a larger one, both passes are charged to the task.
- The quality floor is set relative to the strongest model in the run (e.g. within 2 percentage points of its accuracy).

## Measurement

- **Energy source:** NVML's cumulative energy counter (`nvmlDeviceGetTotalEnergyConsumption`), read before and after each request or batch. On A100/H100 this is more reliable than integrating sampled power draw.
- **Scope:** GPU only. Host CPU, memory, networking and cooling are not included; results are reported as GPU-side energy.
- **Idle power:** reported separately and amortized explicitly, never silently dropped.
- **Hardware is a variable, not a footnote:** every result records GPU model, driver, batch size, sequence length, quantization and serving config.

## Reporting rules

- Report the full quality-vs-energy frontier, not a single best point.
- Publish raw logs and the exact command that produced each number.
- Freeze the eval suite before collecting results; changing it invalidates earlier numbers.

## Open questions

- Best way to amortize idle power across requests at low utilization.
- How to attribute energy per request when requests share a batch.
