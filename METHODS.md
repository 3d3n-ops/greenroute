# Methods (draft)

This is a living document. Critique is welcome — open an issue. [`ROADMAP.md`](ROADMAP.md) says what is being built and why.

## Metrics

**J/task@Q** — energy to complete one task at a fixed quality floor, with every retry, sample and escalation charged to the task.

- A task counts as complete only if it passes the task's automatic check: the exact numeric answer for GSM8K, all unit tests for HumanEval.
- The quality floor is set relative to the best single configuration in the comparison: a policy passes if its accuracy is not more than 1 percentage point below it, tested with a paired bootstrap over the same problems.
- If a request escalates from one model to another, both passes are charged to the task. So are confidence checks and extra samples.

**Tokens per solved task** — output tokens (reasoning included) divided by tasks solved. Measurable for any model, including closed APIs.

**J per token** — GPU energy per output token at a stated load. Measurable only for models we run.

**Power curve** — for each model on each GPU, power is modelled as idle power, plus a serving floor while any request is active, plus a per-token cost that grows with throughput. It is fitted from a server sweep across request rates and used to attribute energy to individual requests, which batched serving cannot measure directly. The first sweep shows power rising gradually at low throughput rather than stepping to the floor, so the straight-line fit overstates low-load power by up to about 25%; until a curved model is validated, low-load numbers come from measured levels, not the fit. Levels where the server falls behind (first-token latency p99 above 2 s) are left out of the fit: past its limit a server is unstable from run to run. The straight line fits the mixture-of-experts gpt-oss models poorly (R² 0.62–0.77): their power rises steeply at low throughput and then levels off. A curved "ramp" model, power = idle + A(1 − e^(−x/s)) + b·x, passes through measured idle power, reduces to the line at high load, and fits every sweep so far within 6 W; its leave-one-out error (refit without each level, then predict it) is 4–24 W against the line's 12–178 W. Prices between measured levels come from the ramp model; it is fitted on 5–7 levels per configuration, so it is checked against new sweeps as they come in.

## Reference loads

Energy per task depends on load more than on anything else, so every energy number states its load. The benchmark reports two:

- **Saturated:** the offline batch run at the highest concurrency (the floor is shared by as many tasks as possible), measured at **steady state** (below), not over the whole run.
- **Latency-bound:** the highest request rate at which first-token latency p99 stays under a stated target, from the server sweep.

Lightly loaded numbers are reported from the power curve, not as a headline.

## Measurement

- **Energy source:** NVML's cumulative energy counter (`nvmlDeviceGetTotalEnergyConsumption`), read before and after each batch or traffic level. On A100/H100 this is more reliable than integrating sampled power draw.
- **Scope:** GPU only, for now. Host CPU, memory, networking and data-centre overhead are added in Phase 3 as stated multipliers, never folded silently into GPU numbers.
- **Idle power:** measured with the model loaded and reported separately; "total" includes it and "active" subtracts it.
- **Hardware is a variable:** every result records GPU model, driver, concurrency, token limits, thinking mode and serving config.
- **Excluded from measurement:** scoring, including running HumanEval tests, happens after the energy span ends.

## Decoding

- **Main answers:** greedy (temperature 0) for Qwen3, as in every earlier run. gpt-oss uses OpenAI's recommended sampling (temperature 1.0, top_p 1.0): under greedy decoding about 1 in 10 of its high-effort answers looped until the token limit. Sampled answers are seeded per request (seed 1,000,000 + request index), so a rerun repeats them.
- **Reasoning settings:** Qwen3 thinking on or off through its chat template; gpt-oss reasoning effort (low, medium, high) through its chat template offline and the `reasoning_effort` request field on the server. gpt-oss replies are parsed from the Harmony format: the answer is the text on the `final` channel, and a reply that never reaches it has no answer.
- **Latency:** first-token time counts the first streamed token of any kind, reasoning included.

## Steady state

An offline batch run ends with a drain-out tail: the last few long answers keep generating while most of the GPU's batch slots sit empty. A live server with continuous traffic would fill those slots, so whole-run energy overstates the saturated cost. On GSM8K with Qwen3 the tail inflated energy per task by 1.04–1.15x in fast mode and 1.4–1.9x in thinking mode, where answer lengths vary most.

- **Timeline:** the engine is driven one decoding step at a time, recording after each step the time, the energy counter, the number of unfinished requests and the output tokens so far.
- **Window:** from the moment one concurrency cap's worth of requests has finished and been replaced (so the batch holds requests of mixed ages, as a live server's does) until unfinished requests fall below the cap (the queue is empty). If that window holds under 10% of the run's output tokens, the window from the start of the run is used instead, and the basis is recorded.
- **Steady J/task:** energy per output token inside the window, times the workload's mean output tokens per task.
- **No steady state:** a run with fewer problems than the concurrency cap never fills the GPU and reports no steady figure rather than a misleading one. HumanEval (164 problems against 256 slots) is therefore run as 8 copies of the set in one batch; copies share the prefix cache, which makes prefill slightly cheaper, and every copy is scored.
- **Sampled answers** are run the same way, one request and seed per sample, so self-consistency gets a steady figure too.
- **Both are reported:** whole-run figures stay in every result file next to the steady ones, along with the tail's energy, time and tokens.

## Measured vs estimated

- Numbers from our own hardware are **measured**.
- Numbers for closed models' energy are **estimated**, always reported as a range, and only from an estimator whose error has been published against measured open models.
- The two are never shown in the same column or combined into one ranking.

## Routing experiments

- Router thresholds and any trained router are fit on GSM8K train problems; test problems are used once, for the reported number.
- Every policy is compared with the best single configuration at the same total traffic, on the same hardware layout, at the same quality floor.
- Results are reported as a curve against traffic level, with the crossover (if any) where routing starts to pay.

## Reporting rules

- Report the full quality-vs-energy frontier, not a single best point.
- Publish raw per-problem logs and the exact command and commit that produced each number. A commit hash ending in `-dirty` means the code was not committed and the number is provisional.
- Run every headline configuration at least three times and report the spread. Measured so far on GSM8K: accuracy moves under 1 point between runs; whole-run energy per task 2–13%, steady-state 1–12%.
- Freeze the task suite before collecting results; changing it invalidates earlier numbers.

## Open questions

- How best to amortize idle power at low utilization, and whether to report it per task or per hour.
- How well the power curve attributes energy to individual requests that share a batch; Phase 4's live runs test this.
- Which grid signal to make primary for decisions: marginal (WattTime) or average (Electricity Maps).
