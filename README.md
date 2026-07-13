# Krauncher

**Run your training script on a remote GPU. Nothing more.**

Krauncher is a minimal Python library for researchers who have a working
local script and need a GPU — not a platform.

Website & API keys: **[krauncher.com](https://krauncher.com)**

---

## Quickstart

```bash
pip install krauncher
export CAS_API_KEY="cas_..."        # krauncher.com → Account → API Keys
```

Requires Python 3.11+.

```python
import asyncio
from krauncher import KrauncherClient

client = KrauncherClient()           # reads CAS_API_KEY / CAS_BROKER_URL from env or .env

@client.task(vram_gb=1, timeout=120)
def multiply(size: int):
    import numpy as np               # imports go INSIDE the function
    a, b = np.random.rand(size, size), np.random.rand(size, size)
    return {"mean": float((a @ b).mean())}

async def main():
    handle = await multiply(size=1000)   # submit → TaskHandle
    print("task:", handle.task_id)
    result = await handle                # await the handle → TaskResult
    print("output:", result.output)
    print("gpu:", result.actual_gpu, "·", f"{result.execution_time_sec:.1f}s")

asyncio.run(main())
```

The decorated function becomes **async**: calling it submits the task and
returns a `TaskHandle`; awaiting the handle (or `await handle.wait(...)`)
returns a `TaskResult`.

> **Using an LLM / coding agent?** Read **[AGENTS.md](AGENTS.md)** — a single
> accurate reference of the API, parameters, result fields, errors and
> constraints. Runnable examples live in **[tutorial/](tutorial/)**.

---

## The problem with serverless ML platforms

Serverless orchestration platforms are genuinely impressive pieces of
infrastructure. They handle container builds, secret management, artifact
storage, scheduling, persistent volumes, and team dashboards.

They also charge you for all of it — whether you use it or not.

If you're fine-tuning a small model, running ablations, or iterating on
a research experiment with a dataset under 2 GB, you're likely paying for
an orchestration layer you don't need.

Krauncher does less, on purpose. It runs your existing Python function on
a remote GPU, returns the result, and gets out of the way.

---

## What Krauncher is (and isn't)

**Good fit:**
- Fine-tuning, LoRA, small-scale experiments with training datasets up to ~2 GB
- Researchers who already have a working local script
- Anyone tired of rewriting their code to fit a platform's abstractions
- Teams where "infrastructure" means one person and a credit card

**Not the right tool if:**
- You need managed versioned artifact storage
- Your team requires persistent shared volumes across runs
- Your dataset is hundreds of GBs with complex multi-node sharding
- You want a UI dashboard for experiment tracking

---

## How it works

Add a decorator. Await your function. Get a result. Your existing code
doesn't change — no base images, no volume mounts, no platform imports.

```python
import asyncio
from krauncher import KrauncherClient

client = KrauncherClient()

@client.task(gpu_name="RTX4090", group_id="mistral-run", timeout=3600)
def finetune():
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    from datasets import load_dataset

    # Weights download to worker storage on first run (~15 GB for 7B);
    # later runs in the same group_id reuse the cached weights.
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

    # ... your training logic, unchanged from local ...

    model.save_pretrained("/tmp/output")
    # Worker storage is ephemeral — sync checkpoints out before returning.
    upload_to_s3("/tmp/output", "my-checkpoints/run-1")
    return {"status": "done", "checkpoint": "s3://my-checkpoints/run-1"}

async def main():
    result = await finetune()        # submit and wait
    print(result.output)

asyncio.run(main())
```

> The decorated function is **async** — always call it from an `async`
> context and `await` the handle (which submits and waits). See the
> [Quickstart](#quickstart) for the canonical shape.

### Choosing a GPU

| Decorator argument          | Effect                                                        |
|-----------------------------|--------------------------------------------------------------|
| `vram_gb=24`                | Require at least 24 GB VRAM                                   |
| `gpu_name="H100"`           | Require a specific model (case-insensitive substring)        |
| `gpu_arch="Ada"`            | Require a GPU architecture                                    |
| *(omit `vram_gb`)*          | **Auto-classify**: the analyzer inspects your code and picks the VRAM tier for you |

Leaving `vram_gb` unset is the recommended default — Krauncher analyzes your
code statically and sizes the GPU automatically.

---

## Security model

Krauncher doesn't store anything. Your API key and training code are
encrypted on your machine before leaving it, and decrypted only inside the
ephemeral worker. The relay that routes your jobs cannot read the payload —
it doesn't have the keys.

| What                       | Visible to Krauncher |
|----------------------------|----------------------|
| Your storage credentials   | No                   |
| Your training code         | No                   |
| Your model weights/outputs | No                   |
| Job timing and GPU type    | Yes                  |

This isn't a feature we added. It's a consequence of not wanting to be in
the data custody business. E2E encryption is on by default (`CAS_ENCRYPT=true`).

---

## Data locality

Tasks with the same `group_id` are routed to the same physical host, so
whatever your first run downloaded to local NVMe is still there for the next.

```python
@client.task(gpu_name="RTX4090", group_id="my-experiment-v1")
def train_epoch(epoch: int):
    import os
    cache_path = "/tmp/dataset.bin"
    if not os.path.exists(cache_path):
        download_from_s3("my-bucket", "dataset.bin", cache_path)
        # subsequent tasks in this group skip this step
    run_training(cache_path, epoch=epoch)
    return {"epoch": epoch, "status": "complete"}

async def main():
    for epoch in range(10):
        await train_epoch(epoch=epoch)
```

For larger or registered datasets, use the **data bridge** (`data_urls=` /
`data=`), which downloads into `/data` inside the sandbox — see
[tutorial/06](tutorial/06_data_bridge.py) and
[tutorial/15](tutorial/15_data_sources_s3.py).

---

## Beyond a single function

- **Notebook / editor cells.** `await client.run_code(code, inputs={...},
  outputs=[...])` runs a code *string* instead of a decorated function: named
  local values go in, named variables come back (JSON-safe, 16 MB budget). This
  is the primitive the `krauncher-jupyter` `%%krauncher` magic is built on. See
  [tutorial/50](tutorial/50_run_code_values.py).
- **Multi-phase runs.** `group = await client.group(task_a, task_b)` derives a
  shared-requirements envelope (VRAM floor, GPU pins, disk) from the tasks and
  keeps them on one warm worker; submit with `await group.submit(task, ...)`.
  See [tutorial/52](tutorial/52_group_envelope.py).

---

## Inspecting a finished task

After a task completes, the broker keeps a structured record — the same one
the web UI renders on the task detail page.

```python
task   = await client.get_task(task_id)         # what GET /tasks/{id} returns
report = await client.get_task_report(task_id)  # task + extended report
```

`get_task` returns status, timing breakdown (queue / download / pip / setup /
execution), classification, costs, GPU and worker specs, and the result.

`get_task_report` adds an extended `report` field: peak/average GPU
utilization, peak VRAM, the actual GPU's hardware specs, and an estimated
time/cost comparison across all known GPUs at the worker's measured host
capabilities. It is intended as feedback for an LLM author of the user
code — pure data, no interpretation.

---

## Examples

Numbered, runnable tutorials in [`tutorial/`](tutorial/):

| #   | File                              | Demonstrates                                  |
|-----|-----------------------------------|-----------------------------------------------|
| 01  | `01_remote_simple.py`             | Minimal submit + await                        |
| 02  | `02_remote_with_deps.py`          | `pip=` dependencies in the sandbox            |
| 03  | `03_error_handling.py`            | Catching `TaskError` / remote tracebacks      |
| 04  | `04_timeout.py`                   | Execution timeout behaviour                   |
| 05  | `05_task_groups.py`               | `group_id` host affinity                      |
| 06  | `06_data_bridge.py`               | `data_urls=` downloads into `/data`           |
| 09  | `09_streaming_logs.py`            | Live logs via `wait(on_log=...)`              |
| 10  | `10_progress_bar.py`              | Progress reporting                            |
| 11  | `11_e2e_encryption.py`            | End-to-end encryption                         |
| 12  | `12_helper_functions.py`          | Shipping helper functions with the task       |
| 13  | `13_bert_finetune.py`             | Real ML code → analyzer classification        |
| 15  | `15_data_sources_s3.py`           | Registered S3 data sources                    |
| 17  | `17_multiphase_training.py`       | Multi-phase training in one group             |
| 18  | `18_resnet152_food101.py`         | ResNet-152 on Food-101                        |
| 19  | `19_huggingface_dataset.py`       | HuggingFace dataset bridge                    |
| 20  | `20_bert_imdb.py`                 | BERT fine-tuning on IMDB                       |
| 21  | `21_qwen25_7b_lora_alpaca.py`     | Qwen2.5-7B LoRA fine-tuning                    |
| 22  | `22_qwen25_7b_inference_gsm8k.py` | Qwen2.5-7B inference                           |
| 23  | `23_gnn_node_classification_cora.py` | GCN node classification                    |
| 30+ | `30_…`–`36_…`                     | LLM inference and batched inference           |
| 50  | `50_run_code_values.py`           | `run_code` with named in/out values           |
| 52  | `52_group_envelope.py`            | `client.group()` multi-phase envelope         |
| 53  | `53_hf_native.py`                 | HuggingFace-native auto pre-fetch             |

---

## Install

```bash
pip install krauncher
export CAS_API_KEY="your_api_key"
```

Requires Python 3.11+.

---

## License

MIT
