# Roadmap

GreenRoute has two goals that feed each other:

1. **A benchmark** that measures how much energy models, open and closed, need to get tasks *right*, and how that changes with load, reasoning effort and hardware.
2. **A serving framework and router** that uses those measurements to serve a workload with the least energy at a given accuracy and latency, and shows what that means for the electricity grid.

The benchmark comes first because every routing decision needs its numbers. [`METHODS.md`](METHODS.md) defines the metrics and rules; this page says what gets built, in what order, and why.

Both lead to a product, described under [Where this leads](#where-this-leads-the-product): a model gateway that routes every request for energy, speed and accuracy together, so people get powerful AI that is sustainable in its use of electricity.

## What we already know (Sprints 0–2)

- **Load dominates.** On one H100, GPU energy per GSM8K task falls 30–43x from 1 request per second to saturation. An H100 with a model loaded draws about 120 W doing nothing.
- **Power has three parts.** Idle (about 120 W), an extra floor as soon as the model serves anything (about 120 W for Qwen3-1.7B, 250 W for Qwen3-8B), and a small per-token cost (about 8 and 15 mJ per output token). A typical answer's own marginal energy is 2–4 J; the rest of the 5–400 J per task is the floor shared over whatever traffic arrives.
- **Routing up a model size is weak on GSM8K.** Escalating 1.7B answers to the 8B by answer length or run-to-run instability needs about 60% escalation to match the 4B alone; Phase 1 confirmed no 1.7B → 8B cascade beats the 4B alone at saturation.

## What Phase 1 has found so far

- **Reasoning effort is the biggest per-request lever, and its price depends on load.** On a full GPU, thinking mode costs 8–11x the energy of a direct answer on GSM8K for 1.5–10 points of accuracy, and 70–100x on HumanEval for 2–10 points. Under light server traffic the GPU's fixed power is paid either way, and thinking costs only 1.4–2.2x.
- **Power curves exist for every Qwen3 model and mode.** Idle about 125 W, a serving floor of 225–386 W, and 10–31 mJ per output token; two runs each agree within a few percent.
- **Effort routing works with a free signal.** Answering with Qwen3-8B fast and escalating its least confident answers to thinking matches always-thinking accuracy (95.1%) at 47% of the energy, with thresholds set on train and scored on test. A perfect router would reach 96.8% at 28%, so a better signal is worth looking for.
- **Offline batches need steady-state measurement.** Whole-run energy overstated thinking mode by 1.4–1.9x because of a drain-out tail; [`METHODS.md`](METHODS.md#steady-state) defines the steady-state window now used.
- **Results are reproducible.** Across repeat runs accuracy moves under 1 point and steady-state energy 1–12%. Server and offline measurements of saturated thinking agree within 7–16%.
- **Self-consistency is expensive.** Four sampled answers cost about 4.5x the answer itself, so it pays only as a signal for problems where a cheaper one fails.
- **A small mixture-of-experts model at low effort beats every Qwen3 configuration on code.** gpt-oss-20b at low effort passes 92.5% of HumanEval at 13.3 J per task; Qwen3-8B thinking passes 88.0% at 827 J. High effort costs 4.7–11x low effort for 1–3 points.
- **Decoding settings can dominate energy.** With greedy decoding about 1 in 10 gpt-oss high-effort answers looped to the token limit, holding over half the tokens. Each family now runs with its recommended decoding.
- **The router's bottleneck is the signal, not the headroom.** A perfect router over gpt-oss settings would use about 5x less energy than the best single setting at higher accuracy, but free signals (answer confidence, length) capture 1.1–2.5x at best. Models that reason before answering are confident in their final answers, so a better signal (reasoning-token confidence, a cheap verifier, task type) is the main open problem.
- **A curved power model works.** Power ramps up from idle and then grows linearly; this fits every sweep within 6 W and prices left-out traffic levels within 4–24 W.
- **Fitting the GPU to the model matters.** gpt-oss-120b on one H100 has little memory left for in-progress answers, so each token costs 2.5–6x the 20b's. Multi-GPU measurement is needed before judging large models.

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

### Phase 1 — Open-model ground truth (done)

A per-problem table for open models, which is both the benchmark's measured core and the router's training data.

- **Configurations:** Qwen3 1.7B, 4B and 8B, each with thinking off and on; at least one model from another family.
- **Tasks with automatic checking:** GSM8K (maths, exact answer) and HumanEval (code, unit tests). LLM-judged tasks are excluded.
- **Per problem:** correctness, prompt and output tokens, whether the answer was cut off, token-probability confidence, and extra sampled answers for self-consistency.
- **Energy:** measured at saturation for every configuration; the power curve against load (floor plus per-token cost) measured with the Sprint 2 server sweep for each model.
- **Routing split:** a slice of GSM8K train is run too, so router thresholds are tuned off the test set.

**Done when:** the table exists for every configuration on both tasks, and each model has a fitted power curve.

**Status:** done on 2026-11-11. The table exists for all six Qwen3 configurations on both tasks, the GSM8K train slice, and Phi-4-mini and OLMo-2 in fast mode. Both tasks have steady-state energy at saturation (HumanEval by running the set 8 times in one batch), sampled answers have steady-state energy, and each Qwen3 model and mode has a power curve from two server sweeps.

**Extension, same day:** gpt-oss-20b and -120b at low, medium and high effort, both tasks at saturation, with server sweeps. Left open: repeat runs for gpt-oss, a 2-GPU check of gpt-oss-120b (code ready, run blocked by the Modal plan), a better routing signal, power curves for Phi-4-mini and OLMo-2, HumanEval repeats for Qwen3, and multi-GPU measurement for larger models.

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

## Where this leads: the product

The end goal is a model gateway, similar in shape to OpenRouter, that gives people one API to open and closed models and routes every request for **energy efficiency, low latency and high throughput together**, at an accuracy level the user chooses. Every response says how much energy it used and how sure we are of that number.

### Why these goals do not conflict

Sprint 2 found that keeping GPUs busy cuts energy per task 30–43x, and that latency stays flat until close to the GPU's limit. Busy GPUs give both higher throughput and lower energy, so for most traffic there is nothing to trade. The real trade-off appears only near saturation, where the router holds a latency target.

### What it does

- **Chooses the model and the effort per request.** Fast answers by default; reasoning ("thinking") only when a signal says the fast answer is probably wrong. Phase 1 shows this can match always-reasoning accuracy at 47% of the energy on GSM8K, and a perfect signal would cut that to about 28%.
- **Offers accuracy modes.** "Efficient", "balanced" and "maximum quality", each with a published accuracy floor per task type, backed by the benchmark.
- **Chooses where and when.** Among providers and regions by measured (open models) or validated estimated (closed models) efficiency; for work that can wait, a batch tier that runs when and where the grid is cleanest.
- **Keeps its own GPUs busy.** Traffic for open models is consolidated onto as few GPUs as the latency target allows.
- **Gives an energy receipt with every response.** Estimated energy and emissions, labelled *measured* or *estimated (range)*. Closed models are always estimated; if the estimator is not good enough, they are labelled unknown instead.

### Two kinds of capacity

| | Routing to other providers | Our own serving pool |
| --- | --- | --- |
| What it covers | Breadth: many models, closed ones included | A few open models with steady traffic |
| Levers we control | Model, effort, provider, region, timing | All of those, plus how busy each GPU is, power caps and response to grid stress |
| Energy figures | Estimated, as ranges | Measured per request |

How busy a GPU is matters most, and only the own pool controls it. So the product starts as a router with a small own pool, and the pool grows only as traffic justifies it: an idle reserved GPU is exactly the waste this project measures.

### How we get hardware

Most providers do not own their GPUs. OpenRouter owns none and earns a fee on credit purchases; Modal rents capacity from many clouds and earns its margin by keeping it busy; Inference.net began by pooling idle GPUs. We follow the same path:

1. **Research (now):** serverless GPUs (Modal), paid only while measuring, stretched with cloud startup credits, NVIDIA Inception, academic compute partnerships and climate-focused grants.
2. **Phase 3:** short bare-metal rentals, because whole-server power measurement needs access to the machine's own power monitoring or a metered power strip.
3. **Pilot:** route to outside providers, plus a few reserved GPUs for two or three open models.
4. **Scale:** long-term leases from GPU cloud operators before any purchase. Power, not chips, is now the scarcer resource, so partners are chosen by their electricity: clean supply and participation in grid-flexibility programmes such as EPRI's DCFlex. That makes "sustainable" a measured claim, not marketing.

### First version

An OpenAI-compatible endpoint serving two or three open models on our own GPUs, with effort routing and an energy receipt on every response, measured live. It tests the whole chain on real traffic before outside providers are added.

### What the research must prove first

| Question | Answered by |
| --- | --- |
| Is there a routing signal cheap enough to capture most of the perfect-router saving? Today's best affordable signal captures only part of it. | Phase 1 and router work |
| Can closed-model energy be estimated within a defensible error? | Phase 2 |
| When and where is electricity cleanest, and how much work can wait? | Phase 3 |
| Does simulated routing energy match live measurement within about 10%? | Phase 4 |
| What do real traffic levels and task mixes look like? | The first version |

### Principles

- Measured and estimated numbers are never mixed, and every energy claim has a published method and raw data behind it.
- Outside verification before any public sustainability claim.
- Report total energy, not only savings per request: cheaper AI tends to mean more AI, and the product should make that visible rather than hide it.

## Risks we are watching

- **Credibility:** measured and estimated numbers stay apart; every number has a committed command and raw log; no funding that could look like influence.
- **Noise:** identical runs differ by up to about 18% in energy per task, so every comparison needs repeats and stated error bars.
- **Saturation of easy tasks:** GSM8K is near its ceiling for 8B models and may be in training data. A harder task is added if routing differences vanish.
- **Rebound:** cheaper AI tends to mean more AI. The benchmark cannot prevent that and the write-up will say so.
