# Copyright (c) 2024-2026 Ilya Sergeev. Licensed under the MIT License.

"""Tutorial 18 (fast): ResNet-152 on Food-101 — quick GPU smoke test.

Same model and dataset as tutorial 18, but limited to 3 epochs × 150 batches
so it finishes in ~10 minutes instead of hours.

Prerequisites:
    1. Register data source in Web UI (Data Sources page):

       "food-101"
         - URLs: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
         - Size: 5 GB
         - is_output: OFF
         - Credentials: (none — public URL)

    2. Seed API key:   python seed_api_key.py
    3. Start services: broker, worker, relay
    4. Configure:      cas-client/.env
    5. Run: python tutorial/18_fast_resnet152_food101.py
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

INPUT_SOURCE = "food-101"
TASK_TIMEOUT = 900


@client.task(
    vram_gb=12,
    timeout=TASK_TIMEOUT,
    pip=[],
    data=INPUT_SOURCE,
)
def train_resnet152(epochs: int, batch_size: int, lr: float, max_batches: int = 0):
    """Train ResNet-152 on Food-101 from scratch."""
    import os
    import tarfile
    import time

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet152

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Unpack dataset ──
    data_root = "/data/food-101"
    archive = "/data/food-101.tar.gz"
    if not os.path.isdir(data_root) and os.path.isfile(archive):
        print("Extracting dataset...")
        t0 = time.time()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall("/data")
        print(f"Extracted in {time.time() - t0:.1f}s")

    # ── Build train split from meta/train.txt ──
    train_dir = os.path.join(data_root, "images")
    transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    n_batches = len(train_loader) if max_batches <= 0 else min(max_batches, len(train_loader))
    print(f"Training samples: {len(train_ds)}, batches/epoch: {n_batches}")

    # ── Model ──
    model = resnet152(num_classes=101).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader, 1):
            if i > n_batches:
                break
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if i % 50 == 0 or i == n_batches:
                print(
                    f"Epoch {epoch}/{epochs}  "
                    f"batch {i}/{n_batches}  "
                    f"loss={running_loss / i:.4f}  "
                    f"acc={correct / total:.3f}",
                    flush=True,
                )

        avg_loss = running_loss / n_batches
        accuracy = correct / total
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})
        print(
            f"Epoch {epoch}/{epochs}  "
            f"loss={avg_loss:.4f}  acc={accuracy:.3f}",
            flush=True,
        )

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_vram:.2f} GB")

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "final_loss": round(history[-1]["loss"], 4),
        "final_accuracy": round(history[-1]["accuracy"], 4),
    }


def on_log(msg: dict) -> None:
    t = msg.get("type")
    d = msg.get("data", {})
    if t == "stdout":
        print(f"  {d.get('text', '').rstrip()}")
    elif t == "event":
        name = d.get("name", "")
        if name not in ("stream_ended",):
            print(f"  [{name}]")
    elif t == "metric":
        gpu = d.get("gpu_util_pct", 0)
        vram = d.get("vram_used_gb", 0)
        print(f"  [gpu] {gpu:.0f}%  VRAM {vram:.1f} GB")


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print(f"Input source: {INPUT_SOURCE}")
    print(f"Model:        ResNet-152 (58M params)")
    print(f"Dataset:      Food-101 (101 classes, ~75k train images)")
    print(f"Mode:         FAST — 3 epochs × 150 batches")
    print()

    h = await train_resnet152(epochs=3, batch_size=64, lr=0.01, max_batches=150)
    print(f"Task ID: {h.task_id}")
    c = h.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB, method={c.analysis_method}, confidence={c.confidence}")
    print("-" * 60)

    r = await h.wait(on_log=on_log, timeout=TASK_TIMEOUT)

    print("-" * 60)
    print(f"Result:")
    print(f"  Epochs:     {r.output['epochs']}")
    print(f"  Batch size: {r.output['batch_size']}")
    print(f"  Loss:       {r.output['final_loss']}")
    print(f"  Accuracy:   {r.output['final_accuracy']}")
    print(f"  Worker:     {r.worker_id}  GPU: {r.actual_gpu}")
    print(f"  Time:       {r.execution_time_sec:.2f}s")
    cur = r.billing_currency
    print(f"  Actual CU:      {r.actual_cu:.4f}")
    print(f"  Provider cost:  {r.provider_cost:.6f} {cur}")
    print(f"  Charged KU:     {r.charged_ku:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
