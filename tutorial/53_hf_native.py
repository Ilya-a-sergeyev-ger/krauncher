"""Tutorial 53: HuggingFace the native way — auto pre-fetch of Hub references.

The code block loads MNIST with a plain ``load_dataset("ylecun/mnist")`` —
no data_urls, no mount paths (compare with tutorial 19). The client side
translates the established notebook practice into data-bridge plumbing:

    1. detect_hf_refs() finds literal Hub references in the block's AST
    2. hf_size_mb() sizes them via the Hub API — the quote gets an honest
       cu_io / disk estimate before anything is submitted
    3. the data bridge pre-fetches the repo into the worker's HF hub cache
       (#layout=cache) BEFORE the container starts — the unmodified
       load_dataset() call finds it through HF_HOME, and the download IO
       stays in the measured download phase instead of polluting compute

The %%krauncher magic runs these steps automatically; this script shows the
raw primitives it is built on.
"""

import asyncio

from krauncher import KrauncherClient
from krauncher.hf import CACHE_FRAGMENT, detect_hf_refs, hf_size_mb
from krauncher.values import decode_outputs

client = KrauncherClient()

TRAIN_CODE = """
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Native practice: no paths — the pre-fetched hub cache serves this call.
ds = load_dataset("ylecun/mnist")
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
    print(f"Epoch {epoch + 1}/{num_epochs}, loss={total_loss / len(train_loader):.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
train_samples = len(train_data)
test_samples = len(test_data)
print(f"Test accuracy: {accuracy:.4f}")
"""

OUTPUTS = ["accuracy", "train_samples", "test_samples"]


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env")
        return

    # Detect and size the Hub references — before anything is submitted.
    hf_urls, dynamic = detect_hf_refs(TRAIN_CODE)
    for d in dynamic:
        print(f"WARNING: {d}: dynamic HF reference — downloads in-code")
    size_mb = await hf_size_mb(hf_urls)
    print(f"HF pre-fetch: {', '.join(u.removeprefix('hf://') for u in hf_urls)}"
          + (f" ({size_mb:.0f} MB)" if size_mb else ""))

    handle = await client.run_code(
        TRAIN_CODE,
        inputs={"num_epochs": 1, "batch_size": 128},
        outputs=OUTPUTS,
        data_urls=[u + CACHE_FRAGMENT for u in hf_urls],
        dataset_size=size_mb,
        pip=["datasets"],  # torch is pre-installed in cas-sandbox
        timeout=300,
    )
    print(f"Task ID: {handle.task_id}")
    c = handle.classification
    print(f"Classification: tier={c.tier}, vram={c.min_vram_gb}GB, CU={c.compute_units}, method={c.analysis_method}")

    if client.estimate_only:
        return  # dry run

    result = await handle.wait()
    values = decode_outputs(result.output, OUTPUTS)

    print(f"Accuracy: {values['accuracy']:.4f} "
          f"({values['train_samples']} train / {values['test_samples']} test)")
    print(f"Worker: {result.worker_id}  GPU: {result.actual_gpu}")
    print(f"HF download (pre-container): {result.download_sec:.2f}s  "
          f"Training: {result.execution_time_sec:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
