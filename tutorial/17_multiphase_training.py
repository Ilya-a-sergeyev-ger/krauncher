# Copyright (c) 2024-2026 Ilya Sergeev. Licensed under the MIT License.

"""Tutorial 17: Multi-phase training with group affinity and data reuse.

Demonstrates a realistic ML workflow where two tasks in the same group
train a ResNet-18 on Tiny ImageNet in two phases.  Because both tasks
share a group_id, the second task runs on the same worker and reuses the
data directory of the first task (instant rename, zero re-download).

Phase 1 (warmup):
    - Downloads tiny-imagenet-200.tar.gz (~237 MB) from S3 via Data Bridge
    - Trains ResNet-18 for 3 epochs
    - Saves checkpoint to /output → synced to S3

Phase 2 (continue):
    - Data is already on disk (group affinity reuse, download ≈ 0 s)
    - Loads checkpoint from /data (saved by phase 1) or falls back to S3
    - Trains 3 more epochs
    - Saves final model to /output → synced to S3

Features demonstrated:
    1. Group affinity — both tasks on the same worker
    2. Data Bridge — 237 MB download from S3
    3. Data reuse — second task skips download entirely
    4. Output upload — checkpoint + final model → S3
    5. Checkpoint continuity — phase 2 resumes from phase 1 weights
    6. Real-time streaming — epoch progress via cas-relay

Data preparation:
    Download Tiny ImageNet, repack as .tar.gz, upload to your S3 bucket:

        wget https://cs231n.stanford.edu/tiny-imagenet-200.zip
        unzip tiny-imagenet-200.zip
        tar czf tiny-imagenet-200.tar.gz tiny-imagenet-200/
        # Upload tiny-imagenet-200.tar.gz to your S3 bucket

Prerequisites:
    1. Register two data sources in Web UI (Data Sources page):

       a) "tiny-imagenet"
          - URLs: s3://your-bucket/tiny-imagenet-200.tar.gz
          - Size: 0.25 GB
          - is_output: OFF
          - Credentials: your S3 access key / secret key

       b) "upload-folder"
          - URLs: (empty or s3://your-bucket/checkpoints/)
          - Size: 1 GB
          - is_output: ON

    2. Seed API key:   python seed_api_key.py
    3. Start services: broker, worker, relay
    4. Configure:      cas-client/.env
    5. Run: python tutorial/17_multiphase_training.py
"""

import asyncio
import uuid

from krauncher import KrauncherClient

client = KrauncherClient()

GROUP = f"train-{uuid.uuid4().hex[:8]}"
INPUT_SOURCE = "tiny-imagenet"
OUTPUT_SOURCE = "upload-folder"

TASK_TIMEOUT = 6000  # seconds — both worker execution and client wait


# ---------------------------------------------------------------------------
# Phase 1: Initial training
# ---------------------------------------------------------------------------

@client.task(
    vram_gb=1,
    timeout=TASK_TIMEOUT,
    group_id=GROUP,
    pip=[],  # torch, torchvision are pre-installed in cas-sandbox
    data=INPUT_SOURCE,
    output=OUTPUT_SOURCE,
)
def train_phase1(epochs: int, batch_size: int, lr: float, max_batches: int = 0):
    """Train ResNet-18 on Tiny ImageNet for the first N epochs."""
    import os
    import tarfile
    import time

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Unpack dataset if needed ──
    data_root = "/data/tiny-imagenet-200"
    archive = "/data/tiny-imagenet-200.tar.gz"
    if not os.path.isdir(data_root) and os.path.isfile(archive):
        print("Extracting dataset...")
        t0 = time.time()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall("/data")
        print(f"Extracted in {time.time() - t0:.1f}s")

    # ── Data loaders ──
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(64, padding=4),
        T.ToTensor(),
        T.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    ])
    train_ds = ImageFolder(os.path.join(data_root, "train"), transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda",
    )
    n_batches = len(train_loader) if max_batches <= 0 else min(max_batches, len(train_loader))
    print(f"Training samples: {len(train_ds)}, batches/epoch: {n_batches}")

    # ── Model ──
    model = resnet18(num_classes=200).to(device)
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

    # ── Save checkpoint to /output (synced to S3) ──
    os.makedirs("/output", exist_ok=True)
    ckpt = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    ckpt_name = "checkpoint.pt"
    torch.save(ckpt, f"/output/{ckpt_name}")
    print(f"Checkpoint saved to /output/{ckpt_name}")

    # Also save a copy in /data so phase 2 finds it after rename
    torch.save(ckpt, f"/data/{ckpt_name}")
    print(f"Checkpoint copied to /data/{ckpt_name}")

    return {
        "phase": 1,
        "epochs": epochs,
        "final_loss": round(history[-1]["loss"], 4),
        "final_accuracy": round(history[-1]["accuracy"], 4),
    }


# ---------------------------------------------------------------------------
# Phase 2: Continue training from checkpoint
# ---------------------------------------------------------------------------

@client.task(
    vram_gb=1,
    timeout=TASK_TIMEOUT,
    group_id=GROUP,
    pip=[],  # torch, torchvision are pre-installed in cas-sandbox
    data=INPUT_SOURCE,
    output=OUTPUT_SOURCE,
)
def train_phase2(epochs: int, batch_size: int, lr: float, max_batches: int = 0):
    """Continue training ResNet-18 from the checkpoint left by phase 1."""
    import os
    import tarfile
    import time

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Check what phase 1 left on disk ──
    data_files = os.listdir("/data")
    print(f"Files in /data: {data_files}")

    data_root = "/data/tiny-imagenet-200"
    archive = "/data/tiny-imagenet-200.tar.gz"

    has_unpacked = os.path.isdir(data_root)
    has_archive = os.path.isfile(archive)
    ckpt_name = "checkpoint.pt"
    has_checkpoint = os.path.isfile(f"/data/{ckpt_name}")

    print(f"Unpacked dataset: {'found' if has_unpacked else 'NOT found'}")
    print(f"Archive:          {'found' if has_archive else 'NOT found'}")
    print(f"Checkpoint:       {'found' if has_checkpoint else 'NOT found'}")

    if has_unpacked:
        print("Dataset already unpacked (data reuse from phase 1)")
    elif has_archive:
        print("Extracting dataset (fallback)...")
        t0 = time.time()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall("/data")
        print(f"Extracted in {time.time() - t0:.1f}s")
    else:
        raise RuntimeError("No dataset found in /data — phase 1 data not available")

    # ── Data loaders ──
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(64, padding=4),
        T.ToTensor(),
        T.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    ])
    train_ds = ImageFolder(os.path.join(data_root, "train"), transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda",
    )

    # ── Model + checkpoint ──
    model = resnet18(num_classes=200).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # Try to load checkpoint from /data (left by phase 1 via group reuse)
    ckpt_path = f"/data/{ckpt_name}"
    start_epoch = 0
    history = []
    if os.path.isfile(ckpt_path):
        print(f"Loading checkpoint from disk: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}, previous accuracy: {history[-1]['accuracy']:.3f}")
    else:
        print("WARNING: No checkpoint found on disk, training from scratch")

    # ── Training loop (continue) ──
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        n_batches = len(train_loader) if max_batches <= 0 else min(max_batches, len(train_loader))
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
                    f"Epoch {epoch}/{start_epoch + epochs}  "
                    f"batch {i}/{n_batches}  "
                    f"loss={running_loss / i:.4f}  "
                    f"acc={correct / total:.3f}",
                    flush=True,
                )

        avg_loss = running_loss / n_batches
        accuracy = correct / total
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})
        print(
            f"Epoch {epoch}/{start_epoch + epochs}  "
            f"loss={avg_loss:.4f}  acc={accuracy:.3f}",
            flush=True,
        )

    # ── Save final model to /output ──
    os.makedirs("/output", exist_ok=True)
    torch.save({
        "epoch": start_epoch + epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, "/output/model_final.pt")
    print("Final model saved to /output/model_final.pt")

    total_epochs = start_epoch + epochs
    return {
        "phase": 2,
        "resumed_from_epoch": start_epoch,
        "total_epochs": total_epochs,
        "final_loss": round(history[-1]["loss"], 4),
        "final_accuracy": round(history[-1]["accuracy"], 4),
        "initial_accuracy": round(history[0]["accuracy"], 4) if history else None,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def on_log(msg: dict) -> None:
    """Relay log callback — prints stdout/events in real time."""
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

    print(f"Group:           {GROUP}")
    print(f"Input source:    {INPUT_SOURCE}")
    print(f"Output source:   {OUTPUT_SOURCE}")
    print()

    # ── Phase 1 ──
    print("=" * 60)
    print("PHASE 1: Initial training (3 epochs)")
    print("=" * 60)
    h1 = await train_phase1(epochs=3, batch_size=128, lr=0.01)
    print(f"Task ID: {h1.task_id}")
    c = h1.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB, method={c.analysis_method}")
    print("-" * 60)

    r1 = await h1.wait(on_log=on_log, timeout=TASK_TIMEOUT)

    print("-" * 60)
    print(f"Phase 1 result:")
    print(f"  Epochs:    {r1.output['epochs']}")
    print(f"  Loss:      {r1.output['final_loss']}")
    print(f"  Accuracy:  {r1.output['final_accuracy']}")
    print(f"  Worker:    {r1.worker_id}")
    print(f"  GPU:       {r1.actual_gpu}")
    print(f"  Download:  {r1.download_sec:.2f}s")
    print(f"  Total:     {r1.execution_time_sec:.2f}s")
    print()

    # ── Phase 2 ──
    print("=" * 60)
    print("PHASE 2: Continue training (3 more epochs)")
    print("=" * 60)
    h2 = await train_phase2(epochs=3, batch_size=128, lr=0.005)
    print(f"Task ID: {h2.task_id}")
    print("-" * 60)

    r2 = await h2.wait(on_log=on_log, timeout=TASK_TIMEOUT)

    print("-" * 60)
    print(f"Phase 2 result:")
    print(f"  Resumed from epoch: {r2.output['resumed_from_epoch']}")
    print(f"  Total epochs:       {r2.output['total_epochs']}")
    print(f"  Loss:               {r2.output['final_loss']}")
    print(f"  Accuracy:           {r2.output['final_accuracy']}")
    print(f"  Worker:             {r2.worker_id}")
    print(f"  GPU:                {r2.actual_gpu}")
    print(f"  Download:           {r2.download_sec:.2f}s")
    print(f"  Total:              {r2.execution_time_sec:.2f}s")
    print()

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    same = r1.worker_id == r2.worker_id
    print(f"  Host affinity:    {'confirmed' if same else 'FAILED'} ({r1.worker_id})")
    print(f"  Data reuse:       phase1 download={r1.download_sec:.2f}s, "
          f"phase2 download={r2.download_sec:.2f}s")
    savings = r1.download_sec - r2.download_sec
    if savings > 0:
        print(f"  Time saved:       {savings:.2f}s (data reuse)")
    print(f"  Accuracy:         {r1.output['final_accuracy']} → {r2.output['final_accuracy']} "
          f"(+{r2.output['final_accuracy'] - r1.output['final_accuracy']:.4f})")
    print(f"  Total CU:         {r1.actual_cu + r2.actual_cu:.4f}")
    print()
    print("Checkpoint and final model are in your S3 output bucket.")


if __name__ == "__main__":
    asyncio.run(main())
