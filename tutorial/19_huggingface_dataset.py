"""Tutorial 19: HuggingFace Dataset — loading data from HuggingFace Hub.

Demonstrates hf:// URL scheme in data_urls: the worker downloads a HuggingFace
dataset *before* container startup and mounts it at /data (read-only).

The function loads the MNIST dataset from /data/, trains a simple CNN for
1 epoch, and returns accuracy on the test split.

Data flow:
    1. Client sends data_urls=["hf://datasets/ylecun/mnist"] in the task request
    2. Worker's DataBridge recognises hf:// scheme, calls huggingface_hub.snapshot_download()
    3. Downloaded repo is mounted into the container at /data/ylecun__mnist/
    4. User code loads data from /data/ylecun__mnist/
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

GPU_NAME="4090"

TIMEOUT=300

@client.task(
    vram_gb=4,
    timeout=TIMEOUT,
#    gpu_name=GPU_NAME,
    data_urls=["hf://datasets/ylecun/mnist"],
    pip=["datasets"],  # torch, torchvision are pre-installed in cas-sandbox
    dataset_size=11,  # MNIST ~11 MB
)
def train_mnist_cnn(num_epochs: int = 1, batch_size: int = 128):
    """Train a simple CNN on MNIST from a pre-downloaded HuggingFace dataset."""
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import os
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset from local files (pre-downloaded by DataBridge)
    _ds_path = "/data/ylecun__mnist"
    _ds_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(_ds_path) for f in fn) / (1 << 20)
    print(f"Loading dataset ({_ds_mb:.0f} MB)...")
    ds = load_dataset("ylecun/mnist", cache_dir=_ds_path)
    train_data = ds["train"]
    test_data = ds["test"]


    def collate(batch):
        images = torch.stack([
            torch.from_numpy(np.array(x["image"], dtype=np.float32)).unsqueeze(0) / 255.0
            for x in batch
        ])
        labels = torch.tensor([x["label"] for x in batch])
        return images, labels

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate)

    # Simple CNN
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "epochs": num_epochs,
        "batch_size": batch_size,
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting MNIST CNN training with HuggingFace dataset...")
    handle = await train_mnist_cnn()
    print(f"Task submitted: {handle.task_id}")

    print("Waiting for result (HF download + training)...")
    result = await handle

    output = result.output
    print(f"\nResults:")
    print(f"  Train samples: {output['train_samples']}")
    print(f"  Test samples:  {output['test_samples']}")
    print(f"  Epochs:        {output['epochs']}")
    print(f"  Accuracy:      {output['accuracy']:.4f}")

    exec_sec = result.execution_time_sec - result.download_sec - result.pip_install_sec
    print(f"\nTiming Breakdown:")
    print(f"  Queue wait:   {result.queue_wait_sec:.2f}s")
    print(f"  HF Download:  {result.download_sec:.2f}s")
    print(f"  Pip install:  {result.pip_install_sec:.2f}s")
    print(f"  Training:     {exec_sec:.2f}s")
    print(f"  Total:        {result.execution_time_sec:.2f}s")
    print(f"  Actual CU:    {result.actual_cu:.4f}")
    print(f"  Charged KU:   {result.charged_ku:.4f}")

    assert output["accuracy"] > 0.95, f"Expected >95% accuracy, got {output['accuracy']:.2%}"
    print("\nAll assertions passed!")


if __name__ == "__main__":
    asyncio.run(main())
