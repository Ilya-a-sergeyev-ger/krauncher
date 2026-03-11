"""Tutorial 13: BERT fine-tuning — tests analyzer classification on real ML code.

Submits a BERT fine-tuning task to verify that cas-analyzer correctly detects:
  - framework: pytorch
  - model: BERT Base (110M params)
  - mode: training
  - precision: fp16
  - optimizer: adamw
  - batch_size, num_epochs

Prerequisites: same as tutorial 10.
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()


@client.task(timeout=300, pip=["torch", "transformers", "datasets", "accelerate"])
def finetune_bert(num_epochs: int, batch_size: int, max_length: int):
    """Fine-tune BERT-base on a small text classification dataset."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, torch_dtype=torch.float16,
    ).to(device)

    # Synthetic dataset
    texts = [
        "This movie was absolutely fantastic and thrilling",
        "Terrible waste of time, would not recommend",
        "Great acting and wonderful cinematography",
        "Boring plot with no character development",
    ] * 64  # 256 samples
    labels = [1, 0, 1, 0] * 64

    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = num_epochs * (len(texts) // batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )

    model.train()
    losses = []
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = len(texts) // batch_size

        for i in range(n_batches):
            s = i * batch_size
            e = s + batch_size
            batch = {k: v[s:e].to(device) for k, v in encodings.items()}
            batch_labels = torch.tensor(labels[s:e], device=device)

            outputs = model(**batch, labels=batch_labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            print(f"\rEpoch {epoch}/{num_epochs}  batch {i+1}/{n_batches}  loss={loss.item():.4f}", end="")

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"\rEpoch {epoch}/{num_epochs}  avg_loss={avg_loss:.4f}          ")

    return {
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "epochs": num_epochs,
        "device": str(device),
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env")
        return

    print("Submitting BERT fine-tuning task...")
    handle = await finetune_bert(num_epochs=3, batch_size=16, max_length=128)
    print(f"Task ID: {handle.task_id}")
    c = handle.classification
    print(f"Classification: tier={c.tier}, vram={c.min_vram_gb}GB, CU={c.compute_units}, method={c.analysis_method}, confidence={c.confidence}")

    result = await handle
    print(f"Result: {result.output}")
    print(f"Worker: {result.worker_id}  GPU: {result.actual_gpu}  Time: {result.execution_time_sec:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
