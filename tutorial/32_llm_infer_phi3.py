"""Tutorial 32: Phi-3-Mini-4K-Instruct (3.8B) inference, 500-token gen.

Calibration target bucket (llm_inference, medium msc, medium dps, small cps,
one epoch, tiny samples). Covers the 1-4B parameter range that sits between
Qwen-1.5B (task 30) and Qwen-7B (task 22 / 31).

Expected runtime: ~1-2 min on RTX 4090, 50 samples × 500 tokens.
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

HF_DATASET = "hf://datasets/openai/gsm8k"
HF_MODEL = "hf://models/microsoft/Phi-3-mini-4k-instruct"


@client.task(
    vram_gb=12,
    timeout=900,
    data_urls=[HF_DATASET, HF_MODEL],
    pip=["datasets"],
    dataset_size=3,
    disk_gb=25,
    stream_stderr=True,
)
def phi3_inference(num_samples: int = 50, max_new_tokens: int = 500):
    """Mid-size LLM Q/A generation."""
    print("Task started. Importing torch / transformers (~12-20s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "/data/microsoft__Phi-3-mini-4k-instruct"
    dataset_path = "/data/openai__gsm8k"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded in {time.monotonic() - t0:.1f}s. "
          f"Loading model weights (fp16, ~7.6 GB)...", flush=True)

    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded in {time.monotonic() - t1:.1f}s.", flush=True)

    ds_full = load_dataset(dataset_path, "main", split="test")
    ds = ds_full.select(range(min(num_samples, len(ds_full))))
    print(f"Running inference on {len(ds)} samples, max_new_tokens={max_new_tokens}",
          flush=True)

    last_log = time.monotonic()
    HEARTBEAT_SEC = 40

    for i, sample in enumerate(ds):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve step by step: {sample['question']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        now = time.monotonic()
        if (i + 1) % 10 == 0 or (now - last_log) >= HEARTBEAT_SEC:
            print(f"  [{i + 1}/{len(ds)}] elapsed {now - t0:.0f}s", flush=True)
            last_log = now

    return {
        "samples": len(ds),
        "max_new_tokens": max_new_tokens,
        "total_tokens": len(ds) * max_new_tokens,
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting Phi-3-Mini inference (mid-size LLM)...")
    handle = await phi3_inference()
    print(f"Task submitted: {handle.task_id}")

    result = await handle.wait(timeout=1200)
    out = result.output
    print(f"\nSamples: {out['samples']}, total tokens: {out['total_tokens']}")
    exec_sec = result.execution_time_sec - result.download_sec - result.pip_install_sec
    print(f"Inference: {exec_sec:.1f}s, Total: {result.execution_time_sec:.1f}s, "
          f"Actual CU: {result.actual_cu:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
