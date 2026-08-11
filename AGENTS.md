# Krauncher — reference for coding agents / LLMs

Accurate, self-contained reference for generating working `krauncher` code.
If anything here conflicts with prose in the README, trust this file and the
docstrings in `krauncher/`. Runnable, verified examples live in `tutorial/`.

`krauncher` runs a plain Python function on a remote GPU and returns its
result. It is **async**. No platform abstractions, no container definitions.

---

## The one canonical pattern

```python
import asyncio
from krauncher import KrauncherClient

client = KrauncherClient()                 # config from env / .env

@client.task(vram_gb=24, timeout=600)      # decorator makes the function async
def train(epochs: int = 3):
    import torch                           # ALL imports go inside the function
    # ... work on the GPU ...
    return {"loss": 0.01}                  # return value must be JSON-serializable

async def main():
    handle = await train(epochs=5)         # calling submits → returns TaskHandle
    print(handle.task_id, handle.classification.tier)
    result = await handle                  # awaiting the handle → TaskResult
    # or: result = await handle.wait(timeout=3600, on_log=print)
    print(result.output, result.actual_gpu, result.charged_ku)

asyncio.run(main())
```

Rules that must hold for generated code:
- The task function is called with **keyword arguments only** (`train(epochs=5)`,
  not `train(5)`).
- The function must be **self-contained**: every import and helper it uses must
  be defined inside it (or passed via the helper-function mechanism, tutorial 12).
  It runs in a fresh sandbox with no access to your module-level globals.
- The return value is serialized — return JSON-compatible data (dict, list,
  str, number, bool, None).
- Everything is async — call task functions inside an `async def` and `await`.

---

## Install & configuration

```bash
pip install krauncher
export CAS_API_KEY="cas_..."
```

Python 3.11+. All settings read from constructor args (highest priority), then
env vars / a `.env` file in CWD.

| Constructor arg     | Env var                | Default                  | Meaning                                   |
|---------------------|------------------------|--------------------------|-------------------------------------------|
| `api_key`           | `CAS_API_KEY`          | — (required)             | API key (`cas_...`)                       |
| `broker_url`        | `CAS_BROKER_URL`       | `https://krauncher.com/api` | Broker base URL                        |
| `analyzer_timeout`  | `CAS_ANALYZER_TIMEOUT` | `10.0`                   | Analyzer call timeout (s)                 |
| `gpu_name`          | `KRAUNCHER_GPU_NAME`   | `""`                     | Default GPU model filter                  |
| `gpu_arch`          | `KRAUNCHER_GPU_ARCH`   | `""`                     | Default GPU arch filter                   |
| `estimate_only`     | `CAS_ESTIMATE_ONLY`    | `false`                  | Run analyzer, return classification, skip submission |
| `stream_stderr`     | `CAS_STREAM_STDERR`    | `false`                  | Stream worker stderr to client            |
| `max_task_retries`  | `CAS_MAX_TASK_RETRIES` | `3`                      | Transparent resubmits after an infrastructure failure (0 disables) |
| `max_task_chain_sec`| `CAS_MAX_TASK_CHAIN_SEC` | `0`                    | Wall-clock ceiling on a retry chain; `0` = twice the task's `timeout` |
| `send_credentials`  | `CAS_SEND_CREDENTIALS` | `true`                   | Attach your storage keys to the encrypted payload (see below) |
| —                   | `KRAUNCHER_VRAM_GB`    | `""`                     | Overrides `@task(vram_gb=...)` — re-targets existing tasks to another VRAM class without editing them |
| —                   | `KRAUNCHER_VRAM_HEADROOM` | `1.05`                | Safety factor on every VRAM requirement, explicit and auto-classified. Values below 1.0 are ignored |
| —                   | `CAS_CLIENT_CONFIG`    | `.env` in CWD            | Path to the config file to load; must be a real env var, not a key inside that file |
| —                   | `KRAUNCHER_DEBUG`      | `false`                  | Verbose client logging                    |

**Task E2E encryption is mandatory** — code and arguments are always encrypted
to the worker, the broker rejects plaintext submissions, and there is no opt-out
switch. The `/estimate` call to the analyzer is also always E2E-encrypted, with
no plaintext fallback.

**Storage credentials go to the worker, not to us.** The keys a task needs for
its own S3 bucket or a private HuggingFace repo are read from your environment
(`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, optional `AWS_REGION` /
`AWS_ENDPOINT_URL`; `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`) and travel sealed
inside the same encrypted payload as the code — the broker never sees them, and
nothing is stored. Partial S3 credentials are skipped rather than sent.
`CAS_SEND_CREDENTIALS=false` attaches none at all.

Relay transport (`KRAUNCHER_RELAY_TLS`, `KRAUNCHER_RELAY_CA`,
`KRAUNCHER_RELAY_AUTHORITY`) is negotiated automatically — the broker
distributes the CA in-memory. Set these only against a self-hosted relay.

---

## `@client.task(...)` parameters

All keyword-only.

| Parameter      | Type         | Default | Meaning                                                                 |
|----------------|--------------|---------|-------------------------------------------------------------------------|
| `vram_gb`      | `int`        | `None`  | Minimum GPU VRAM. **`None` = auto-classify** the code via the analyzer.  |
| `gpu_name`     | `str`        | `None`  | Required GPU model, case-insensitive substring (`"H100"`, `"L4"`). `""` = no filter; `None` = client default. |
| `gpu_arch`     | `str`        | `None`  | Required GPU architecture (`"Ada"`). `""` = no filter; `None` = client default. |
| `pip`          | `list[str]`  | `None`  | Pip packages installed in the sandbox before execution.                 |
| `timeout`      | `int`        | `600`   | Remote execution timeout (seconds). Worker kills the task past this → `RemoteTimeout`. |
| `priority`     | `int`        | `1`     | 0 = highest, 10 = lowest.                                                |
| `data_urls`    | `list[str]`  | `None`  | URLs (incl. `hf://...`, `s3://...`) downloaded into `/data`.             |
| `data`         | `str`        | `None`  | Registered data source name; broker resolves URLs + creds → `/data`.    |
| `output`       | `str`        | `None`  | Registered output data source; task writes to `/output`, broker uploads. |
| `volume`       | `str`        | `None`  | Persistent volume name; synced to `/volume` before and after.           |
| `group_id`     | `str`        | `None`  | Host affinity — same `group_id` ⇒ same worker (warm caches).            |
| `provider`     | `str`        | `None`  | Pin to a provider (`"runpod"`, `"local"`). `None` = cheapest suitable.   |
| `disk_gb`      | `int`        | `10`    | Minimum disk (GB). Broker takes max of this and data-source size.       |
| `dataset_size` | `float`      | `None`  | Dataset size (MB) for CU estimation; overrides auto-resolved size.      |
| `stream_stderr`| `bool`       | `None`  | Per-task override of client `stream_stderr`.                            |
| `artifacts`    | `bool`       | `False` | Return the files the task wrote beside itself → `result.artifacts`.     |

**GPU selection:** prefer leaving `vram_gb=None` (auto-classify). Use
`gpu_name`/`gpu_arch`/`vram_gb` only to constrain.

---

## `TaskHandle`

Returned by calling a task function.

- `handle.task_id: str`
- `handle.classification` — `TaskClassification` (`.tier`, `.min_vram_gb`,
  `.compute_units`, `.confidence`, `.analysis_method`); `None` if not classified.
- `await handle` → `TaskResult` (shorthand for `handle.wait()`).
- `await handle.wait(*, timeout=600.0, on_log=None)` → `TaskResult`.
  - `timeout: float` — client-side wait limit; exceeding raises `TaskTimeout`.
  - `on_log: Callable[[dict], None]` — called per live relay message
    (`{"type": "stdout"|"stderr"|"event"|"metric", ...}`). Passing `print` is fine.

---

## `TaskResult` fields

```
task_id: str           status: str            worker_id: str
output: Any            # the function's return value
stdout: str            stderr: str            traceback: str | None
exit_code: int         actual_gpu: str
execution_time_sec: float   duration_sec: float   gpu_util_avg: float
queue_wait_sec: float       download_sec: float   pip_install_sec: float
artifacts: dict[str, bytes] | None   # files the task wrote; see below
# Billing (client currency unless noted):
actual_cu: float            # measured compute units
provider_cost: float        # raw provider cost, no markup/fee
charged_ku: float           # compute charge in KU (markup, no fee)
charged_local: float        fee_ku: float         fee_local: float
total_charged_ku: float     # charged_ku + fee_ku (full balance deduction)
total_charged_local: float  billing_currency: str
```

---

## Files in, files out

Values return as JSON; files do not. Both directions ride the same encrypted
package that already carries the code, so there is no storage to configure and
no mount path to remember — and the broker sees none of it.

```python
@client.task(vram_gb=1, timeout=300, artifacts=True)   # opt in to get files back
def transform(width=160):
    from PIL import Image
    text = open("sample.txt").read()                   # arrived beside the task
    open("out.txt", "w").write(text.upper())           # written beside the task
    Image.new("RGB", (width, 120)).save("gradient.png")
    return {"ok": True}

handle = await transform(width=160, files={"sample.txt": b"hello"})
result = await handle
result.files                    # ["gradient.png", "out.txt"] — names, sorted
result.artifacts["out.txt"]     # b"HELLO" — raw bytes, never base64
result.download("received")     # write them under ./received, returns the count
```

- `files={name: bytes}` is a **call-time** argument, not a `@client.task`
  option — it travels beside the code, not as a task argument. A task that
  declares a parameter named `files` raises `KrauncherError` at decoration.
- Files arrive in the task's working directory, which is also its `HOME`. Files
  the task writes there come back only when `artifacts=True`.
- Hidden files and directories are skipped — they are caches libraries drop in
  `~`, not task output.
- Both directions share the **16 MB inline budget** with the code and the
  result. For a dataset or a checkpoint use `volume=`; overshooting is reported
  before the wire, not after.
- `result.artifacts is None` means the task declared none, or the worker did not
  act on the declaration; `{}` means it was handled and the task wrote nothing.
- `download()` resolves names against the destination and refuses any that
  escape it — a result cannot write elsewhere on your disk. Tutorial 54.

---

## Analysis and execution are separate phases

Every submission is analysis (classify the code, price it) then execution
(encrypt, submit, run). They can be split:

| Want | Call |
|---|---|
| Classify a code block, run it later without re-analysis | `estimate_code(code, ...)` → `run_code(code, ..., classification=...)` |
| Size a whole sequence before submitting anything | `await client.group(task_a, task_b)` |
| Analyze everything, submit nothing | `estimate_only=True` / `CAS_ESTIMATE_ONLY` |
| Per-GPU predicted time and cost, before any client call | `POST /api/estimate` (see below) |

A decorated `@client.task` function classifies once and reuses that result for
every later call, but its classification cannot be passed in or out — reuse
across processes is a `run_code` capability.

---

## Running a code block — `run_code` (values API)

`@client.task` wraps a *function*. `run_code` runs a *code string* (a notebook
cell, an editor selection) — the primitive notebook/editor adapters build on.
Named local values ride in as inputs; named variables from the block's
namespace come back as outputs, through the relay's encrypted result mailbox.

```python
handle = await client.run_code(
    code,                       # source executed as-is on the worker
    inputs={"epochs": 3, "batch_size": 32},   # injected as the block's variables
    outputs=["losses", "accuracy"],           # collected from its namespace → result.output
    lenient_outputs=True,       # drop unset / non-JSON-safe names instead of failing
    pip=["torch"], timeout=300, # any @client.task option also applies
)
result = await handle
losses = result.output["losses"]              # or krauncher.values.decode_outputs(...)
```

- `inputs` / `outputs` values must be **JSON-safe** and fit the **16 MB inline
  budget** shared with the code (`krauncher.values.INLINE_BUDGET_BYTES`); larger
  data goes through a `volume=` / `data=` source. Numeric inputs stay visible to
  the analyzer's CU estimate — don't wrap them.
- Auto-detect the transfer set with `krauncher.codeblock.analyze_names(code)` →
  `(free_vars, assigned_names)`; free variables are inputs, assigned names are
  outputs. This is exactly what the `%%krauncher` magic uses.
- Extra `task_options`: `classification=` — a precomputed `TaskClassification`
  (skips analysis), and `group=` — a `TaskGroup` for warm-worker co-location.
  `artifacts=` / `files=` work here too (see "Files in, files out").

`client.estimate_code(code, *, inputs=, outputs=, lenient_outputs=, vram_gb=,
data=, volume=, dataset_size=)` → `TaskClassification`: classify the exact
source `run_code` would submit, **without** running it. Pass the result to
`run_code(..., classification=...)` to execute without a second analysis.

### HuggingFace-native pre-fetch

Literal `load_dataset("org/name")` / `from_pretrained("org/name")` references in
a block are detected (`krauncher.hf.detect_hf_refs`) and sized
(`hf.hf_size_mb`) so the quote is honest, then pre-fetched into the worker's HF
cache **before the container starts** — the unmodified call finds them via
`HF_HOME`, with no `data_urls=` needed. Only *literal* refs are pre-fetched;
f-string / variable refs download inside execution.

---

## Jupyter cell magic — `%%krauncher`

A separate package (`krauncher-jupyter`, repo `cas-jupyter`) built on
`run_code`: it marks one notebook cell to run remotely while the kernel stays
local. Documented here because agents writing notebook code need the flags.

```python
%pip install krauncher-jupyter     # pulls this SDK
%load_ext krauncher_magic
```

```python
%%krauncher --pip torch
model = build_model().to("cuda")
losses = train(model, epochs, batch_size)   # epochs, batch_size come from the notebook
accuracy = evaluate(model)
```

No flags are required: the cell's free variables become inputs and its
assigned names become outputs (`codeblock.analyze_names`), so `losses` and
`accuracy` are ordinary notebook variables afterwards. The price is quoted
before the cell runs; logs stream live.

| Flag | Meaning |
|---|---|
| `--in NAMES` | override the auto-detected inputs (comma-separated, repeatable) |
| `--out NAMES` | override the auto-detected outputs |
| `--pip PKGS` | pip packages installed in the sandbox before execution |
| `--vram N` | minimum GPU VRAM in GB (default: auto-classified from the code) |
| `--gpu-name S` | pin a GPU model (case-insensitive substring, e.g. `A4000`) |
| `--timeout N` | execution timeout in seconds (default 600) |
| `--dataset-size MB` | declared input size for the quote (e.g. private S3 objects) |
| `--async [NAME]` | non-blocking: inject a task handle as `NAME` (default `kr_task`) |
| `--estimate` | classify and print the quote only — do not run |

Same credentials as the SDK (`CAS_API_KEY`, optional `CAS_BROKER_URL`), same
JSON-safe 16 MB inline budget for transferred values, and no GPU state across
cells — each marked cell is its own ephemeral task.

---

## Task groups — `client.group()`

A `TaskGroup` is a shared-requirements envelope derived from the member tasks
themselves — for a sequence that should share one warm worker (Tier-1 group
affinity), replacing hand-picked `group_id` strings duplicated on every task.

```python
group = await client.group(train_phase1, train_phase2)   # analysis only, nothing submitted
h1 = await group.submit(train_phase1, epochs=3)           # or run_code(..., group=group)
r1 = await h1.wait()
h2 = await group.submit(train_phase2, epochs=3)           # reuses phase-1 worker warm
```

The envelope is: VRAM floor = max over members (explicit `vram_gb` pins get the
usual VRAM headroom); `gpu_name` / `gpu_arch` / `provider` shared — **conflicting
explicit pins raise**; disk = max member `disk_gb` + total size of all members'
data sources / volumes. Fields: `group.group_id`, `group.vram_floor`,
`group.disk_gb`. Tutorial 52.

---

## Volumes & registered data sources

Handles created off the client; see the `volume=` / `data=` / `output=` task
params for how a task mounts them.

```python
vol = client.volume("my-vol", size_gb=5)   # ensure exists → Volume handle
vol.upload("local/dir", "dest"); vol.ls("prefix"); vol.download("remote", "local"); vol.info()

src = client.data_source("my-data", urls=["s3://bucket/key"], size_gb=2)  # register
out = client.data_source("out", urls=["s3://bucket/out"], is_output=True) # upload target
src.info(); src.delete()
```

`client.list_runners(print_table=True)` → `list[Runner]` — current fleet
(`GET /admin/fleet`), grouped by provider; prints a table by default (handy in
notebooks).

---

## Inspecting a finished task

```python
task   = await client.get_task(task_id)        # dict, == GET /tasks/{id}
report = await client.get_task_report(task_id) # {**task, "report": {...}}
```

`get_task_report` adds `report`: peak/avg GPU util, peak VRAM, the actual
GPU's specs, and a per-GPU time/cost comparison — pure data for an LLM to
reason about optimizing the user code.

---

## Pre-run cost estimate (HTTP) — for optimization loops

Get the predicted **per-GPU time and cost** for a task *before submitting it*,
priced against the live cross-provider market. This is the signal an LLM/agent
optimizes against. It is an HTTP endpoint (no client method yet); `estimate_only`
returns only the classification, `get_task_report` returns prices only *after* a
run — this returns the priced list up front.

```
POST https://krauncher.com/api/estimate
  X-API-Key: cas_...          # optional; anonymous is IP-limited
  {"code": "<source containing a @client.task function>"}
```

Response (rows sorted cheapest-first by `estimated_cost_usd`):

```jsonc
{
  "rows": [
    { "gpu_name": "RTX 5060 Ti", "vram_gb": 16,
      "estimated_sec": 356.18, "estimated_cost_usd": 0.009747,
      "min_price_usd": 0.0985, "prices": { "vastai": 0.0985 } }
  ],
  "cu_breakdown": { "cu_compute": 0.0, "cu_io": 0.0, "cu_setup": 0.0 }, // where the cost lives
  "setup_sec": 0.0, "io_sec": 0.0,                    // measured setup / I/O time
  "provider_warmup_sec": { "vastai": 45 },            // cold start, per provider
  "min_vram_gb": 6, "confidence": 0.0, "analysis_method": "ast"        // "ast" | "llm"
}
```

- `rows[0]` is the cheapest GPU that fits. Empty `rows` ⇒ CPU-only or the model
  wasn't recognized (no GPU compute to price).
- `cu_breakdown` shows whether the job is compute-, I/O-, or setup-bound — what to
  optimize. `provider_warmup_sec` is the cold start that precedes the run; on a
  short task it can dominate the choice of provider.
- A prediction is an **expected** cost, not a guarantee (real-host variance).
- **The loop:** edit the task → estimate → keep what's cheaper → repeat. Costs no
  GPU-seconds; ideal as an agent tool.

---

## MCP server — the estimate as an agent tool

`krauncher-mcp` exposes the pre-run estimate as a single MCP tool, `estimate`,
for agents that speak the Model Context Protocol (Claude Desktop, Claude Code,
Cursor, ...). It wraps the analyzer, not the broker — the code is analyzed
statically, never executed.

```bash
pip install krauncher-mcp
```

```jsonc
// MCP client config — no key needed
{ "mcpServers": { "krauncher-analyzer": { "command": "krauncher-mcp" } } }
```

Without a key it runs **keyless** against the public analyzer, under a per-IP
daily quota (a 429 comes back as a short note to register for a larger one). Set
`KRAUNCHER_API_KEY` to use your account and skip the quota. Override the endpoint
with `KRAUNCHER_ANALYZER_URL` when self-hosting the analyzer.

`estimate(code)` returns the task's cost **profile on the reference card** (RTX
PRO 6000 WS, the card CU is normalized to), not the per-GPU priced lineup that
`POST /api/estimate` returns:

```jsonc
{
  "reference_card": "RTX PRO 6000 WS",
  "compute_sec": 20.2, "setup_sec": 3.0, "io_sec": 2.1,   // the three phases
  "min_vram_gb": 6, "min_disk_gb": 10,                     // what it needs to run
  "confidence": 1.0, "analysis_method": "ast",             // how much to trust it
  "cpu_only": false,
  "spread": 1.51, "spread_reason": "1.51x { cv_training, nw=one } on the compute phase, observed on 42 runs across 16 hosts (worst 2.45x)",
  "calibration_basis": "calibrated",                       // | extrapolated | uncalibrated
  "knobs": [                                               // the run parameters to try
    { "name": "num_workers", "value": null, "same_work": true },
    { "name": "batch_size",  "value": "16", "same_work": true },
    { "name": "num_epochs",  "value": "3",  "same_work": false }
  ],
  "findings": ["batch_size=16", "Recognized model: BERT Base (0.11B params)"]
}
```

The seconds are a **relative signal for comparing code against code** (fixed
reference card), not the wall-clock on the GPU the task ends up on. Use it in the
loop: edit the run → estimate → keep the cheaper variant → repeat, no GPU spent.
For the priced per-GPU lineup, use `POST /api/estimate` above.

`knobs` is the shortlist worth re-estimating, so you do not have to guess which
parameters the estimate responds to. Change their **values only** — not the
model, the architecture, the dataset, or the training procedure. The parameters
that move the time without changing what the code produces are listed every
time, found or not: `value: null` means the analyzer did not see it in the source
(it arrives as a call argument, a config entry or an environment variable), so
name it in the code and estimate again. `same_work: false` (epochs, steps,
sequence length) marks a knob that shrinks the job itself — lowering it buys a
smaller result, not a cheaper one, and belongs in the answer as a change of task
rather than a saving.

`spread` is a measured population factor, not a doubt about the reading: 1.51
means the slow end of the hosts this shape was measured on takes about half
again as long as the estimate, and `spread_reason` names the population and the
run count behind it. Confidence 1.0 with a large spread is a coherent answer —
the code was read correctly, and the GPU is simply not what governs its time:
the card waits on the host, so the run inherits whichever host it lands on. The
lever is GPU utilization — whatever leaves the card idle in this code (data
loaded in the main process, per-item preprocessing, synchronous transfers, a
batch too small to fill the card). `calibration_basis` says whether this shape was measured (`calibrated`),
answered by a neighbouring one (`extrapolated`), or matched nothing at all
(`uncalibrated`, where the seconds are an order of magnitude rather than a
figure).

---

## Exceptions (`from krauncher import ...`)

All inherit `KrauncherError`.

| Exception                  | Raised when                                                        |
|----------------------------|-------------------------------------------------------------------|
| `AuthError`                | 401/403 from the broker (bad/missing API key).                    |
| `InsufficientBalanceError` | 402 — not enough KU. Attrs: `required_ku`, `available_ku`, `balance_ku`, `held_ku`, `predicted_ku`, `fee_ku`. |
| `TaskError`                | Task failed on the worker. Attrs: `task_id`, `remote_traceback`.  |
| `RemoteTimeout`            | Worker killed the task at `timeout` (subclass of `TaskError`).    |
| `PayloadDeliveryError`     | Encrypted payload couldn't reach the worker; **not charged**, retry. |
| `TaskTimeout`              | `handle.wait()` exceeded its client-side `timeout`.               |
| `NoCapacityError`          | No matching hosts; **not charged**.                               |
| `SerializationError`       | The task function couldn't be serialized.                         |

---

## Constraints & gotchas

- **Self-contained function.** No module-level globals, no closures over outer
  variables. Imports and helper defs go inside the function (or use tutorial 12).
- **Async only.** Task calls, `wait`, `get_task`, `get_task_report` are all
  coroutines — `await` them inside `async def`.
- **Keyword args only** when calling the task.
- **Ephemeral worker storage.** `/tmp` and local disk vanish after the task.
  Persist results by returning them, or push to S3 / an `output` source / a
  `volume`. `/data` is read-only input; `/output` and `/volume` persist. Files
  written beside the task come back with `artifacts=True`.
- **Return must serialize** (JSON-compatible). Don't return tensors/models.
- **Datasets:** sweet spot ≤ ~2 GB. Use `data_urls=`/`data=` (→ `/data`) and
  `hf://...` paths for HuggingFace assets (tutorials 06, 15, 19).
- **Warm caches:** reuse one `group_id` across a sequence to keep the worker
  and its downloaded weights/data (tutorials 05, 17).
- **Estimate without running:** `CAS_ESTIMATE_ONLY=true` (or
  `KrauncherClient(estimate_only=True)`) returns the classification and skips
  submission.

---

## Examples (`tutorial/`)

Start at `01_remote_simple.py`. Then by topic: deps `02`, errors `03`, timeout
`04`, groups `05`/`17`, data bridge `06`/`15`/`19`, live logs `09`, progress
`10`, E2E `11`, helper functions `12`, classification on real ML code `13`,
vision training `18`, BERT/IMDB `20`, LoRA `21`, inference `22`/`30`–`34`,
batched inference `35`/`36`. Values / adapters primitives: `run_code` with
named in/out values `50`, `client.group()` envelope `52`, HuggingFace-native
auto pre-fetch `53`, files in / artifacts out `54`.
