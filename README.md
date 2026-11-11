# GreenRoute

An open benchmark for the energy cost of LLM inference — measured per **completed task at a fixed quality bar**, retries included.

> Status: early. Sprint 0 is underway. No results yet.

## The question

How many joules does it take an LLM to actually finish a task (solve a GSM8K problem, pass a HumanEval test) at a given quality level — and how much of that can be cut by choosing the right model, quantization, batch size, time, and place to run it?

Most published numbers are per token or estimated from FLOPs. We measure GPU energy directly and report **J/task@Q**: joules per completed task at a fixed quality floor, with every retry and escalation charged to the task.

## What's here

| Path | What |
| --- | --- |
| [`METHODS.md`](METHODS.md) | How we measure. Read this first if you want to review the work. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to help, and how credit works. |
| `greenroute/` | The measurement and benchmark code (in progress). |
| `results/` | Raw logs and measured numbers, as they come. |

## Roadmap

- **Sprint 0** — one small model served with vLLM, GPU energy instrumented, first J/task number logged.
- **Sprint 1** — one model family across sizes × quantization, frozen eval suite.
- **Sprint 2** — does energy track API price? Router vs best single model at the same quality.

## Help wanted

Reviews of the measurement method are the most useful thing right now. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
