# Krauncher

**Run your training script on a remote GPU. Nothing more.**

Krauncher is a minimal Python library for researchers who have a working
local script and need a GPU — not a platform.

---

## The problem with serverless ML platforms

Serverless orchestration platforms are genuinely impressive pieces of
infrastructure. They handle container builds, secret management, artifact
storage, scheduling, persistent volumes, and team dashboards.

They also charge you for all of it — whether you use it or not.

If you're fine-tuning a small model, running ablations, or iterating on
a research experiment with a dataset under 2 GB, you're likely paying for
an orchestration layer you don't need. The markup over raw GPU cost on
those platforms typically runs 80–150%.

Krauncher does less, on purpose. It runs your existing Python function on
a remote GPU, returns the result, and gets out of the way. Our overhead
is 20–30% over raw provider cost. That's the whole business model.

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

Add a decorator. Run your function. Get a result.

Your existing code doesn't need to change. No base images to define,
no volume mounts to configure, no platform-specific imports.

```python
import krauncher

client = krauncher.Client()

@client.task(gpu="RTX4090")
def finetune():
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    from datasets import load_dataset

    # Model weights are downloaded to worker storage on first run (~15 GB for 7B).
    # Subsequent runs in the same group_id reuse the cached weights.
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

    # ... your training logic, unchanged from local ...

    model.save_pretrained("/tmp/output")
    # Worker storage is ephemeral — sync your checkpoint out before returning.
    upload_to_s3("/tmp/output", "my-checkpoints/run-1")
    return {"status": "done", "checkpoint": "s3://my-checkpoints/run-1"}

result = finetune()
```

---

## Security model

Serverless platforms necessarily see your environment: your secrets,
your code, often your data. That's a reasonable trade-off when you're
buying their full infrastructure stack.

Krauncher doesn't store anything. Your API keys and training code are
encrypted on your machine before leaving it, and decrypted only inside
the ephemeral worker. The relay that routes your jobs cannot read the
payload — it doesn't have the keys.

| What                       | Visible to Krauncher |
|----------------------------|----------------------|
| Your storage credentials   | No                   |
| Your training code         | No                   |
| Your model weights/outputs | No                   |
| Job timing and GPU type    | Yes                  |

This isn't a feature we added. It's a consequence of not wanting to be
in the data custody business.

---

## Pricing

| Overhead over raw GPU cost     |           |
|--------------------------------|-----------|
| Big Orchestration Platforms    | 80–150%   |
| Krauncher                      | 20–30%    |

We broker compute from RunPod and Vast.ai. You can check their public
rates. Our margin is the difference.

---

## Data locality

We don't manage your data. But the scheduler is smart enough to keep
your worker warm: tasks with the same `group_id` are routed to the same
physical host, so whatever your first run downloaded to local NVMe is
still there for the next one.

```python
@client.task(gpu="RTX4090", group_id="my-experiment-v1")
def train_epoch(epoch: int):
    import os

    cache_path = "/tmp/dataset.bin"
    if not os.path.exists(cache_path):
        download_from_s3("my-bucket", "dataset.bin", cache_path)
        # subsequent tasks in this group skip this step

    run_training(cache_path, epoch=epoch)
    return {"epoch": epoch, "status": "complete"}

for epoch in range(10):
    train_epoch(epoch=epoch)
```

---

## Install

```bash
pip install krauncher
export KRAUNCHER_API_KEY="your_api_key"
```

Requires Python 3.9+.

---

## License

MIT
