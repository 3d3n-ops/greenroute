# Roadmap

GreenRoute has two goals that feed each other:

1. **A benchmark** that measures how much energy models, open and closed, need to get tasks *right*, and how that changes with load, reasoning effort and hardware.
2. **A serving framework and router** that uses those measurements to serve a workload with the least energy at a given accuracy and latency, and shows what that means for the electricity grid.

The benchmark comes first because every routing decision needs its numbers. [`METHODS.md`](METHODS.md) defines the metrics and rules; this page says what gets built, in what order, and why.

## What we already know (Sprints 0–2)

- **Load dominates.** On one H100, GPU energy per GSM8K task falls 30–43x from 1 request per second to saturation. An H100 with a model loaded draws about 120 W doing nothing.
- **Power has three parts.** Idle (about 120 W), an extra floor as soon as the model serves anything (about 120 W for Qwen3-1.7B, 250 W for Qwen3-8B), and a small per-token cost (about 8 and 15 mJ per output token). A typical answer's own marginal energy is 2–4 J; the rest of the 5–400 J per task is the floor shared over whatever traffic arrives.
- **Cheap routing signals are weak on GSM8K.** Escalating 1.7B answers to the 8B by answer length or run-to-run instability needs about 60% escalation to match the 4B alone. Token probabilities and real sampling were not recorded and are the next signals to test.

## Where this fits

Existing work each covers part of the picture:

- **ML.ENERGY Leaderboard:** measured GPU energy per response, open models only, not per correct answer.
- **AI Energy Score:** star ratings across tasks; closed models only if the provider submits, and the major labs have not.
- **SemiAnalysis InferenceMAX / InferenceX:** throughput per megawatt by hardware, without accuracy.
- **Jegham et al., "How Hungry is AI?":** closed-model energy estimated from API speed and assumed hardware, never checked against measured truth.

None combines energy per *correct* task, its dependence on load, closed models, and grid conditions. That combination is our contribution.

## The central idea

Energy per solved task splits into two factors:

**J per solved task = tokens per solved task × J per token**

- **Tokens per solved task** is model behaviour. It is measurable exactly for any model, closed ones included, because APIs report token counts. Reasoning length makes it vary by more than 10x between models and modes.
- **J per token** is infrastructure. It is measurable only for open models on hardware we control, and it depends strongly on load. For closed models it must be estimated, and the estimate must be validated.

## Phases

### Phase 1 — Open-model ground truth (in progress)

A per-problem table for open models, which is both the benchmark's measured core and the router's training data.

- **Configurations:** Qwen3 1.7B, 4B and 8B, each with thinking off and on; at least one model from another family.
- **Tasks with automatic checking:** GSM8K (maths, exact answer) and HumanEval (code, unit tests). LLM-judged tasks are excluded.
- **Per problem:** correctness, prompt and output tokens, whether the answer was cut off, token-probability confidence, and extra sampled answers for self-consistency.
- **Energy:** measured at saturation for every configuration; the power curve against load (floor plus per-token cost) measured with the Sprint 2 server sweep for each model.
- **Routing split:** a slice of GSM8K train is run too, so router thresholds are tuned off the test set.

**Done when:** the table exists for every configuration on both tasks, and each model has a fitted power curve.

### Phase 2 — Closed-model behaviour and a validated energy estimator

- Run the same tasks through closed-model APIs: accuracy and tokens per solved task, measured exactly.
- Build an estimator that predicts J per token from what an API reveals (speed, latency, token counts). Test it on Phase 1's open models, where the true energy is known, and publish its error.
- Closed models appear on the leaderboard with **measured** tokens and accuracy, and an **estimated range** for energy, never mixed. If the estimator's error is too large, that is the finding.

### Phase 3 — From GPU joules to the grid

Chain each result through server overhead (CPU, memory, networking), data-centre overhead (PUE), and grid emissions by location and hour:

- **Marginal emissions** (WattTime): what *added* load causes. The right signal for decisions.
- **Average and flow-traced intensity** (Electricity Maps): for reporting and scheduling forecasts.
- **Peak and flexibility:** how much of a workload can move in time, and what moving it saves.

For closed providers we do not know which data centre served a request, so this layer is scenarios, labelled as such. For self-hosted serving it can be exact.

### Phase 4 — Serving framework and router

Given a workload (tasks per day, latency target, accuracy floor, location, how much can wait), recommend a configuration and show the energy and emissions of each option. Levers, in the order the data says they matter:

1. Consolidate traffic onto busy GPUs.
2. Match reasoning effort to the task on one model and one GPU.
3. Route between models where traffic keeps both pools busy.
4. Shift deferrable work to cleaner hours or regions.
5. Shed power during grid stress for flexible work.

Every recommendation is checked against a live measured run; simulated energy must land within about 10% of measured, or the energy model is revised.

## Risks we are watching

- **Credibility:** measured and estimated numbers stay apart; every number has a committed command and raw log; no funding that could look like influence.
- **Noise:** identical runs differ by up to about 18% in energy per task, so every comparison needs repeats and stated error bars.
- **Saturation of easy tasks:** GSM8K is near its ceiling for 8B models and may be in training data. A harder task is added if routing differences vanish.
- **Rebound:** cheaper AI tends to mean more AI. The benchmark cannot prevent that and the write-up will say so.
