# Multi-turn local coding-agent adapter (final-stage use only)

`python -m bench.agent_http` is an executable OpenAI-compatible, five-tool
agent client. Its **default is DRY_RUN**: it reads a registered disposable task,
prints an execution plan, and does not contact an endpoint or execute candidate
code. Its existing fixture and Python tests use a scripted **HTTP mock**, not a
model or Zephyrus inference. The benchmark's target is whole-task correctness
and time, not a benchmark of a single prose completion.

The agent can call `list_files`, `read_file`, `write_file`, `run_tests`, or
`finish` one at a time. The client generates valid OpenAI assistant/tool reply
pairs with linked tool-call IDs. Writes are constrained to a task's allowlisted
source files and a source-byte limit. Tests must follow the latest edit. Generated
source cannot be executed on the host through these tools.

`run_tests` invokes a pinned local Docker image with network disabled, a
read-only mounted disposable workspace, fixed memory/CPU/process limits and
**only the visible test suite**. The model sees a bounded pass/fail/count result,
not test logs or grader metadata. Once the agent finishes, a separate grader
copies the candidate and runs both public and evaluator-owned tests in Docker.
The final grade and file hashes are attached by the client, **not accepted as
model-generated tool arguments**. Noncompletion and malformed model calls are
recorded as failed tasks; grader/infrastructure faults remain unverified.

Docker does **not** provide a hardened hostile-code sandbox, and generated
code executed during final grading can share that grading container with the
private test module. Do not claim protection against a malicious model that
deliberately introspects its execution environment or attempts test gaming.
Only run real candidate code on a disposable unprivileged machine.

## Safe offline checks

```powershell
python -m bench.agent_http --task bench/agent_tasks/energy_rollup_v1
python -m bench.agent_http --task bench/agent_tasks/config_merge_v1
python -m unittest discover -s tests -q
```

## When final-stage validation is explicitly approved

The `--execute` path requires the existing explicit environment authorization,
`--armed-report` from the reviewed strict-cache source, immutable `--controls`
and `--hardware-receipt` JSON, and `--image` pinned by Docker image digest.
The model endpoint must be `http://127.0.0.1:<port>/v1/chat/completions`;
redirects, proxies, arbitrary hosts and unsupported response shapes are rejected.
Use `--arm`, `--repeat` and `--record-output` to collect a matched campaign.
All generated source and raw result files remain under ignored `.local/`.

A successful live run writes one private `.local/benchmark/agent-trials.jsonl`
record compatible with `python -m bench.fitness`. The record includes actual
server-reported input/output token totals, model turn count and whole-task
client wall time. **TTFT and intertoken decode time are `null`**, since this
adapter requests complete, non-streaming tool calls rather than a token stream.
Never turn those unavailable timings into zeroes or an invented token/s rate.
A faster incomplete tool session is a failed task, not a successful benchmark.

There is no automatic model selection, production promotion or GPU launch from
an offline mock result. The first real stage must independently validate actual
weight placement, compiled kernel dispatch, model/tool correctness and free/peak
GPU memory before any performance interpretation. Raw hardware receipts are
operator supplied and must be checked independently: JSON provenance fields
alone are not a cryptographic attestation of which model or GPU was used.
