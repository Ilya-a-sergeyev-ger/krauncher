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
| `broker_url`        | `CAS_BROKER_URL`       | `https://krauncher.com`  | Broker base URL                           |
| `encrypt`           | `CAS_ENCRYPT`          | `true`                   | E2E-encrypt code/args                     |
| `encrypt_analyzer`  | `CAS_ENCRYPT_ANALYZER` | `true`                   | E2E-encrypt code sent to the analyzer     |
| `analyzer_timeout`  | `CAS_ANALYZER_TIMEOUT` | `10.0`                   | Analyzer call timeout (s)                 |
| `gpu_name`          | `KRAUNCHER_GPU_NAME`   | `""`                     | Default GPU model filter                  |
| `gpu_arch`          | `KRAUNCHER_GPU_ARCH`   | `""`                     | Default GPU arch filter                   |
| `estimate_only`     | `CAS_ESTIMATE_ONLY`    | `false`                  | Run analyzer, return classification, skip submission |
| `stream_stderr`     | `CAS_STREAM_STDERR`    | `false`                  | Stream worker stderr to client            |

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
# Billing (client currency unless noted):
actual_cu: float            # measured compute units
provider_cost: float        # raw provider cost, no markup/fee
charged_ku: float           # compute charge in KU (markup, no fee)
charged_local: float        fee_ku: float         fee_local: float
total_charged_ku: float     # charged_ku + fee_ku (full balance deduction)
total_charged_local: float  billing_currency: str
```

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
  `volume`. `/data` is read-only input; `/output` and `/volume` persist.
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
batched inference `35`/`36`.
