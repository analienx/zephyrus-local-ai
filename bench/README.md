# Local AI fitness harness — executable, not a token/s leaderboard

**Status:** offline evaluator, planned balanced campaigns, final-stage-only local streaming collector, and optional code/whole-card telemetry. No Zephyrus model loads, inference, GPU kernel calls or performance benchmarks were run while building it. The bundled case responses and timings are **SIMULATED**, not measured.

## What it can measure and what it cannot yet

- Each trial binds a suite digest, exact case ID/repeat, model and engine revisions, quantization, KV cache, MTP depth, reserved context, sampler, prompt template, output limit and power mode. A/B comparisons reject any undeclared control difference, incomplete case pairing or mixed simulated/measured provenance.
- Critical tool-call, exact-retrieval and structured JSON answers are graded deterministically; a `python_syntax` case **never passes based on syntax alone**. It needs an accompanying behavioral-test receipt for the exact code and suite-owned test source, produced inside a digest-pinned, non-networked, resource-limited Docker image. Docker is **not** a hardened security boundary for arbitrary malicious code; use a disposable unprivileged environment.
- The collector can eventually use an OpenAI-compatible streaming loopback endpoint and *actual server-reported* token usage. It records client-observed first output and completion times; these are **not** GPU prefill/decode kernel measurements. A backend that cannot report token counts fails closed instead of being credited with invented speed.
- Optional future `nvidia-smi` sampling reports whole-card VRAM, GPU temperature, power and clocks (not per-process allocated/reserved CUDA memory). Sampler perturbation, cache reuse, Windows graphics, sustained heat and valid effective kernel dispatch must still be controlled in the final stage.
- Current synthetic smoke tasks prove harness logic, **not model quality**. Before selection, grow the suite with representative repository edits plus real tests, multi-turn tool sessions, long-context retrieval, formatting, refusals, multilingual work, and code tasks from a disposable public fixture repo. Compare task success and end-to-end time first; collect internal target/draft counters and GPU profiler data later.

## Safe commands now (no model or GPU contacted)

```powershell
python -m unittest discover -s tests -v
python -m bench.collector --suite bench/fixtures/smoke-suite.json
python -m bench.fitness --suite bench/fixtures/smoke-suite.json --records bench/fixtures/simulated-trials.jsonl --arm-a A --arm-b B --output .local/benchmark/simulated-report.json
```

The bundled simulated candidate B is faster but fails a critical tool argument; it **must not** qualify as an improvement. Its Python case remains unverified until isolated behavioral tests exist. Raw records and reports belong under ignored `.local/`, never in published benchmarks without review and sanitization.

## Final-stage execution contract — NOT authorized or run yet

`python -m bench.campaign --suite <suite.json> --controls-a <baseline.json> --controls-b <candidate.json> --changed-control <one_control>` only **plans** a five-block A/B/B/A campaign (ten matched repeats per arm). Match output limits and actual input-token counts; random seed, exact prompt, hidden task artifacts and warmup state should also be captured for real work. No single scalar combines task quality, speed and context into a trustworthy universal winner.

Only **after** separate final-stage permission, a reviewed patched source checkout, model weights, endpoint and an `armed` preflight may `bench.collector --execute` contact `http://127.0.0.1:<port>/v1/chat/completions`. It additionally requires `LOCAL_AI_FINAL_BENCHMARKS_AUTHORIZED=I_AUTHORIZE_FINAL_GPU_BENCHMARKS`, explicit controls and revision/hardware receipt JSON. Optional `--docker-image python:3.12-slim@sha256:<digest>` grades suite-owned Python behavioral tests; optional `--gpu-telemetry` samples `nvidia-smi` whole-card statistics. Neither is active in dry-run mode.

An **armed static preflight is not a numerical correctness or VRAM-fit certificate**. The final gate needs independently observed effective kernel dispatch, allocated/reserved process VRAM, warm/cold context, power/fan state, token accounting, true task-test execution and a sustained run. Be explicit about different engines, templates, cache quantization, or CPU offload: those are full-system comparisons, not isolated kernel wins. Compare baseline versus candidate only after the exact settings and provenance are audited.

Synthetic timings in `fixtures/simulated-trials.jsonl` are intentionally arbitrary and must never be presented as laptop performance, including its output-token-rate fields.

## Disposable repository-agent tasks (offline calibration)

Two CPU-only fixture repositories are bundled: `agent_tasks/energy_rollup_v1` and
`agent_tasks/config_merge_v1`. Each ships a deliberately incomplete source,
minimal visible tests, separate evaluator-owned acceptance cases and a trusted
reference implementation **for calibrating the evaluator only**. The agent's
file tools expose **only** the copied `workspace/` directory, never `grader/`.
The grader is published in this public repo, so it is not a secret from someone
who browses the repository; future real comparisons must also use fresh private
variants to reduce benchmark contamination and must prohibit candidate network
access. Do not count these two cases as a comprehensive model-quality evaluation.

```powershell
python -m bench.agent_replay --task bench/agent_tasks/energy_rollup_v1 --transcript bench/fixtures/agent-replay.json
python -m bench.collector --suite bench/fixtures/agent-suite.json
python -m unittest discover -s tests -q
```

The first command rehearses actual list/read/write/test-request/finish tool I/O
in an ignored disposable workspace; it deliberately replies `unverified` to
`run_tests` and does **not** execute candidate Python. `bench.agent_loop`
implements a multi-turn, transport-injected **scripted** session, tested without
an inference endpoint. `bench.repo_judge` prepares an isolated read-only,
network-disabled, pinned-image Docker test run, but refuses execution unless a
separate final-stage caller explicitly authorizes it. Container isolation is
not a hardened boundary for hostile code. `bench.fitness` accepts `repo_task`
results only with matching evaluator-owned public/private test receipts; model
claims or a fast but untested edit remain `unverified`. The existing one-shot
collector explicitly rejects repository tasks in live mode; no misleading
one-turn result is passed off as completed agent work.

## Multi-turn repository agent (offline-ready)

[OpenAI-compatible adapter and separated public/private grading](agent_live_README.md)
connect both registered repo-edit tasks to the existing fitness ledger.
The scripted loopback HTTP integration is tested with no model and no GPU.
The real endpoint remains a separately gated final-stage operation.
