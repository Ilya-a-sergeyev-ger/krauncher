"""Tutorial 33: Qwen2.5-7B-Instruct inference with num_return_sequences=3.

Calibration target bucket (llm_inference, large msc, medium dps, small cps,
one epoch, tiny samples). Open replacement for the planned Llama-3-8B run —
exercises the multi-sequence sampling path (HF generate with
`num_return_sequences > 1` returns a batched output sharing the prompt's
KV cache).

Expected runtime: ~2-3 min on RTX 4090, 30 samples × 256 tokens × 3 seqs.
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

HF_DATASET = "hf://datasets/openai/gsm8k"
HF_MODEL = "hf://models/Qwen/Qwen2.5-7B-Instruct"


@client.task(
    vram_gb=20,
    timeout=7200,
    data_urls=[HF_DATASET, HF_MODEL],
    pip=["datasets"],
    dataset_size=3,
    disk_gb=40,
    stream_stderr=True,
)
def qwen7b_batched_inference(
    num_samples: int = 30,
    max_new_tokens: int = 256,
    num_return_sequences: int = 3,
):
    """Sampling-based generation, 3 candidates per prompt."""
    print("Task started. Importing torch / transformers (~15-25s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s.", flush=True)

    t0 = time.monotonic()
    model_path = "/data/Qwen__Qwen2.5-7B-Instruct"
    dataset_path = "/data/openai__gsm8k"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded in {time.monotonic() - t0:.1f}s. "
          f"Loading model weights (fp16, ~14 GB)...", flush=True)

    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto",
    )
    model.eval()
    print(f"Model loaded in {time.monotonic() - t1:.1f}s.", flush=True)

    ds_full = load_dataset(dataset_path, "main", split="test")
    ds = ds_full.select(range(min(num_samples, len(ds_full))))
    print(f"Running inference on {len(ds)} samples, "
          f"max_new_tokens={max_new_tokens}, num_return_sequences={num_return_sequences}",
          flush=True)

    last_log = time.monotonic()
    HEARTBEAT_SEC = 40

    for i, sample in enumerate(ds):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve briefly: {sample['question']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
            )
        now = time.monotonic()
        if (i + 1) % 5 == 0 or (now - last_log) >= HEARTBEAT_SEC:
            print(f"  [{i + 1}/{len(ds)}] elapsed {now - t0:.0f}s", flush=True)
            last_log = now

    return {
        "samples": len(ds),
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "total_tokens": len(ds) * max_new_tokens * num_return_sequences,
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting Qwen2.5-7B inference (3-sequence sampling)...")
    handle = await qwen7b_batched_inference()
    print(f"Task submitted: {handle.task_id}")

    result = await handle.wait(timeout=7500)
    out = result.output
    print(f"\nSamples: {out['samples']}, total tokens: {out['total_tokens']}")
    exec_sec = result.execution_time_sec - result.download_sec - result.pip_install_sec
    print(f"Inference: {exec_sec:.1f}s, Total: {result.execution_time_sec:.1f}s, "
          f"Actual CU: {result.actual_cu:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
