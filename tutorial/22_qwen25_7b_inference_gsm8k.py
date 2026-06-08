"""Tutorial 22: Qwen2.5-7B-Instruct batched inference on GSM8K.

Runs Qwen/Qwen2.5-7B-Instruct on a subset of openai/gsm8k (grade-school
math word problems), measures answer accuracy. No training — pure
generation. Dataset and model are pre-downloaded via hf:// data bridge.

Calibration target bucket: workload=inference, msc=large (>1B params).
This bucket is currently uncovered in the analyzer's calibration tables.

Expected runtime: ~15-25 min on RTX 4090 (200 samples, max_new_tokens=256).
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

HF_DATASET = "hf://datasets/openai/gsm8k"
HF_MODEL = "hf://models/Qwen/Qwen2.5-7B-Instruct"


@client.task(
    vram_gb=20,
    timeout=4800,
    data_urls=[HF_DATASET, HF_MODEL],
    pip=["datasets"],
    dataset_size=3,  # gsm8k ~3 MB
    disk_gb=40,
)
def qwen_inference_gsm8k(
    num_samples: int = 200,
    max_new_tokens: int = 256,
):
    """Generate solutions for GSM8K math problems with Qwen2.5-7B-Instruct."""
    # Print before imports so the user sees the task is alive even while
    # torch/transformers are loading (~15-25s on a cold container).
    print("Task started. Importing torch / transformers (~15-25s)...",
          flush=True)
    import re
    import time

    _t_imp = time.monotonic()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s. "
          f"Loading tokenizer...", flush=True)

    t0 = time.monotonic()

    model_path = "/data/Qwen__Qwen2.5-7B-Instruct"
    dataset_path = "/data/openai__gsm8k"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded in {time.monotonic() - t0:.1f}s. "
          f"Loading model weights (fp16, ~14 GB)...", flush=True)

    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded in {time.monotonic() - t1:.1f}s. "
          f"Switching to eval mode.", flush=True)
    model.eval()

    print(f"Loading dataset from {dataset_path}...", flush=True)
    ds_full = load_dataset(dataset_path, "main", split="test")
    ds = ds_full.select(range(min(num_samples, len(ds_full))))

    print(f"Running inference on {len(ds)} samples, "
          f"max_new_tokens={max_new_tokens}", flush=True)

    correct = 0
    total = 0
    num_re = re.compile(r"-?\d+\.?\d*")
    last_log = time.monotonic()
    HEARTBEAT_SEC = 50

    for i, sample in enumerate(ds):
        messages = [
            {
                "role": "user",
                "content": (
                    f"Solve this step by step. Put the final numeric answer "
                    f"on the last line.\n\nProblem: {sample['question']}"
                ),
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        gt_raw = sample["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gt = float(gt_raw)
        except ValueError:
            continue

        preds = num_re.findall(response.replace(",", ""))
        if preds:
            try:
                pred = float(preds[-1])
                if abs(pred - gt) < 1e-3:
                    correct += 1
            except ValueError:
                pass

        total += 1

        now = time.monotonic()
        if (i + 1) % 20 == 0 or (now - last_log) >= HEARTBEAT_SEC:
            elapsed = now - t0
            print(f"  [{i + 1}/{len(ds)}] running accuracy: "
                  f"{correct / total:.3f} (elapsed {elapsed:.0f}s)", flush=True)
            last_log = now

    accuracy = correct / total if total else 0.0

    return {
        "samples": len(ds),
        "evaluated": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "max_new_tokens": max_new_tokens,
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting Qwen2.5-7B-Instruct inference on GSM8K subset...")
    print(f"  Dataset: {HF_DATASET}")
    print(f"  Model:   {HF_MODEL}")
    print(f"  Expected runtime: ~15-25 min (download + generation)")
    handle = await qwen_inference_gsm8k()
    print(f"Task submitted: {handle.task_id}")

    def on_log(msg: dict):
        if msg.get("type") not in ("stdout", "stderr"):
            return
        text = (msg.get("data") or {}).get("text") or ""
        for line in text.splitlines():
            low = line.lower()
            if any(k in low for k in (
                "running accuracy", "loading", "loaded",
                "task started", "running inference",
            )):
                print(f"  {line.rstrip()}")

    result = await handle.wait(on_log=on_log, timeout=2400)

    output = result.output
    print(f"\nResults:")
    print(f"  Samples:          {output['samples']}")
    print(f"  Evaluated:        {output['evaluated']}")
    print(f"  Correct:          {output['correct']}")
    print(f"  Accuracy:         {output['accuracy']}")
    print(f"  Max new tokens:   {output['max_new_tokens']}")

    dl_sec = result.download_sec
    exec_sec = result.execution_time_sec - dl_sec - result.pip_install_sec
    print(f"\nTiming Breakdown:")
    print(f"  Queue wait:       {result.queue_wait_sec:.2f}s")
    print(f"  HF Download:      {dl_sec:.2f}s")
    print(f"  Pip install:      {result.pip_install_sec:.2f}s")
    print(f"  Inference:        {exec_sec:.2f}s")
    print(f"  Total:            {result.execution_time_sec:.2f}s")
    print(f"  Actual CU:        {result.actual_cu:.4f}")
    print(f"  Charged KU:       {result.charged_ku:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
