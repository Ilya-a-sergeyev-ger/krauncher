"""Tutorial 21: Qwen2.5-7B LoRA fine-tuning on an Alpaca subset.

Fine-tunes Qwen/Qwen2.5-7B with LoRA adapters on a 3k subset of
tatsu-lab/alpaca using the HuggingFace Trainer API. Dataset and base
model are pre-downloaded via hf:// data bridge.

Calibration target bucket: workload=lora_training, msc=large (>1B params).
This bucket is currently uncovered in the analyzer's calibration tables.

Expected runtime: ~20-30 min on RTX 4090 / A5000 (3 epochs, 3k samples).
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

HF_DATASET = "hf://datasets/tatsu-lab/alpaca"
HF_MODEL = "hf://models/Qwen/Qwen2.5-7B"


@client.task(
    vram_gb=24,
    timeout=4200,
    data_urls=[HF_DATASET, HF_MODEL],
    pip=["peft", "datasets"],
    dataset_size=22,  # alpaca ~22 MB
    disk_gb=40,
    stream_stderr=True,
)
def lora_finetune_qwen_alpaca(
    num_samples: int = 3000,
    num_epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_len: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lr: float = 2e-4,
):
    """LoRA fine-tune Qwen2.5-7B on Alpaca instructions (causal LM)."""
    # Print before imports so the user sees the task is alive even while
    # torch/transformers/peft are loading (~20-30s on a cold container).
    print("Task started. Importing torch / transformers / peft "
          "(~20-30s)...", flush=True)
    import time

    _t_imp = time.monotonic()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    print(f"Imports done in {time.monotonic() - _t_imp:.1f}s. "
          f"Loading tokenizer...", flush=True)

    t0 = time.monotonic()

    model_path = "/data/Qwen__Qwen2.5-7B"
    dataset_path = "/data/tatsu-lab__alpaca"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded in {time.monotonic() - t0:.1f}s. "
          f"Loading base model weights (fp16, ~14 GB)...", flush=True)

    t1 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    print(f"Base model loaded in {time.monotonic() - t1:.1f}s. "
          f"Applying LoRA adapters...", flush=True)

    t2 = time.monotonic()
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f"LoRA applied in {time.monotonic() - t2:.1f}s. "
          f"Loading dataset from {dataset_path}...", flush=True)

    t3 = time.monotonic()
    ds_full = load_dataset(dataset_path)["train"]
    ds = ds_full.shuffle(seed=42).select(range(min(num_samples, len(ds_full))))

    def format_alpaca(ex):
        if ex.get("input"):
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}{tokenizer.eos_token}"
            )
        else:
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n{ex['output']}{tokenizer.eos_token}"
            )
        return tokenizer(text, truncation=True, max_length=max_seq_len, padding="max_length")

    print(f"Dataset loaded ({len(ds)} samples) in {time.monotonic() - t3:.1f}s. "
          f"Tokenizing...", flush=True)
    t4 = time.monotonic()
    ds = ds.map(format_alpaca, remove_columns=ds.column_names)
    ds.set_format("torch")
    print(f"Tokenization complete in {time.monotonic() - t4:.1f}s. "
          f"Train samples: {len(ds)}, max_seq_len={max_seq_len}", flush=True)

    args = TrainingArguments(
        output_dir="/tmp/qwen-lora",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.0,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=2,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class HeartbeatCallback(TrainerCallback):
        """Wall-clock heartbeat: prints at least once per HEARTBEAT_SEC even
        if a single step is slower than logging_steps would emit."""
        HEARTBEAT_SEC = 50

        def __init__(self, t_start: float):
            self._t_start = t_start
            self._last_log = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            now = time.monotonic()
            if now - self._last_log >= self.HEARTBEAT_SEC:
                total = state.max_steps or 0
                elapsed = now - self._t_start
                loss = (state.log_history[-1].get("loss")
                        if state.log_history else None)
                loss_str = f"loss={loss:.4f}" if loss is not None else "loss=n/a"
                print(f"  [step {state.global_step}/{total}] {loss_str} "
                      f"(elapsed {elapsed:.0f}s)", flush=True)
                self._last_log = now

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[HeartbeatCallback(t0)],
    )

    print(
        f"Starting LoRA training: epochs={num_epochs}, "
        f"batch={batch_size}x{grad_accum}, lr={lr}, r={lora_r}",
        flush=True,
    )
    train_result = trainer.train()

    print(f"Training loss: {train_result.training_loss:.4f}", flush=True)

    return {
        "train_loss": round(train_result.training_loss, 4),
        "train_samples": len(ds),
        "epochs": num_epochs,
        "effective_batch": batch_size * grad_accum,
        "max_seq_len": max_seq_len,
        "lora_r": lora_r,
        "train_runtime_sec": round(train_result.metrics["train_runtime"], 1),
        "train_steps": int(train_result.metrics.get("train_steps_per_second", 0) * train_result.metrics["train_runtime"]),
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting Qwen2.5-7B LoRA fine-tune on Alpaca subset...")
    print(f"  Dataset: {HF_DATASET}")
    print(f"  Model:   {HF_MODEL}")
    print(f"  Expected runtime: ~20-30 min (download + training)")
    handle = await lora_finetune_qwen_alpaca()
    print(f"Task submitted: {handle.task_id}")

    def on_log(msg: dict):
        if msg.get("type") not in ("stdout", "stderr"):
            return
        text = (msg.get("data") or {}).get("text") or ""
        for line in text.splitlines():
            low = line.lower()
            if any(k in low for k in (
                "loss", "epoch", "task started", "loaded", "loading",
                "tokenization", "tokenizing", "lora applied",
                "starting lora training", "[step ",
            )) or "%" in line:
                print(f"  {line.rstrip()}")

    result = await handle.wait(on_log=on_log, timeout=4200)

    output = result.output
    print(f"\nResults:")
    print(f"  Train samples:    {output['train_samples']}")
    print(f"  Epochs:           {output['epochs']}")
    print(f"  Effective batch:  {output['effective_batch']}")
    print(f"  Max seq len:      {output['max_seq_len']}")
    print(f"  LoRA rank:        {output['lora_r']}")
    print(f"  Train loss:       {output['train_loss']}")
    print(f"  Train runtime:    {output['train_runtime_sec']}s")

    dl_sec = result.download_sec
    exec_sec = result.execution_time_sec - dl_sec - result.pip_install_sec
    print(f"\nTiming Breakdown:")
    print(f"  Queue wait:       {result.queue_wait_sec:.2f}s")
    print(f"  HF Download:      {dl_sec:.2f}s")
    print(f"  Pip install:      {result.pip_install_sec:.2f}s")
    print(f"  Training:         {exec_sec:.2f}s")
    print(f"  Total:            {result.execution_time_sec:.2f}s")
    print(f"  Actual CU:        {result.actual_cu:.4f}")
    print(f"  Charged KU:       {result.charged_ku:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
