"""Tutorial 20: BERT fine-tuning on IMDB — sentiment classification.

Fine-tunes bert-base-uncased on the full IMDB dataset (25k train / 25k test)
using the HuggingFace Trainer API. Dataset and model are pre-downloaded via
hf:// data bridge — no network access needed during training.

The code is intentionally close to a standard HuggingFace tutorial. The only
CaS-specific parts are the @client.task decorator and local paths for
load_dataset / from_pretrained.

Expected runtime: ~15-20 min on RTX 4090 (3 epochs, full dataset).
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

HF_DATASET = "hf://datasets/stanfordnlp/imdb"
HF_MODEL = "hf://models/google-bert/bert-base-uncased"

GPU_NAME="H100"

@client.task(
    vram_gb=6,
    timeout=1800,
#    gpu_name=GPU_NAME,
    data_urls=[HF_DATASET, HF_MODEL],
    dataset_size=84,  # IMDB dataset ~84 MB
)
def finetune_bert_imdb(num_epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    """Fine-tune BERT on IMDB sentiment classification (positive/negative)."""
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    print("Task started. Waiting for result (download + training, ~15-20 min)...")

    # Load from pre-downloaded local paths (hf:// data bridge)
    model_path = "/data/google-bert__bert-base-uncased"
    dataset_path = "/data/stanfordnlp__imdb"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2,
    )

    import os
    _ds_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(dataset_path) for f in fn) / (1 << 20)
    print(f"Loading dataset ({_ds_mb:.0f} MB)...")
    ds = load_dataset(dataset_path)

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    print("Tokenizing...")
    ds = ds.map(tokenize, batched=True, batch_size=1000)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Training arguments — standard HF Trainer config
    training_args = TrainingArguments(
        output_dir="/tmp/bert-imdb",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": float(accuracy)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"Starting training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
    train_result = trainer.train()

    print("Evaluating...")
    eval_result = trainer.evaluate()

    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Eval accuracy: {eval_result['eval_accuracy']:.4f}")

    return {
        "train_loss": round(train_result.training_loss, 4),
        "eval_accuracy": round(eval_result["eval_accuracy"], 4),
        "eval_loss": round(eval_result["eval_loss"], 4),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_runtime_sec": round(train_result.metrics["train_runtime"], 1),
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting BERT fine-tuning on IMDB (full dataset, 3 epochs)...")
    print(f"  Dataset: {HF_DATASET}")
    print(f"  Model:   {HF_MODEL}")
    print(f"  Expected runtime: ~15-20 min (download + training)")
    handle = await finetune_bert_imdb()
    print(f"Task submitted: {handle.task_id}")

    def on_log(line: str):
        # Show HF Trainer progress logs
        if "loss" in line.lower() or "epoch" in line.lower() or "%" in line:
            print(f"  {line}")

    result = await handle.wait(on_log=on_log, timeout=1800)

    output = result.output
    print(f"\nResults:")
    print(f"  Train samples:   {output['train_samples']}")
    print(f"  Eval samples:    {output['eval_samples']}")
    print(f"  Epochs:          {output['epochs']}")
    print(f"  Train loss:      {output['train_loss']}")
    print(f"  Eval accuracy:   {output['eval_accuracy']}")
    print(f"  Eval loss:       {output['eval_loss']}")
    print(f"  Train runtime:   {output['train_runtime_sec']}s")


    assert output["eval_accuracy"] > 0.85, f"Expected >85% accuracy, got {output['eval_accuracy']:.2%}"
    print("\nAll assertions passed!")


if __name__ == "__main__":
    asyncio.run(main())
