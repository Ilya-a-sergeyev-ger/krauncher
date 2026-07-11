"""Tutorial 50: run_code — send local values to the GPU task and get results back.

The values API executes a code *string* remotely (the same primitive notebook
and editor adapters build on). Named local values ride into the task as plain
JSON kwargs — numeric ones (epochs, batch_size) stay visible to the CU
estimator — and variables computed on the GPU come back through the relay's
encrypted result mailbox: the broker never stores task data.

Based on tutorial 14's custom CNN: training config and a locally generated
validation batch go in; per-epoch losses and predictions come out.
"""

import asyncio
import random

from krauncher import KrauncherClient
from krauncher.values import decode_outputs

client = KrauncherClient()

# The code block runs remotely as-is: `epochs`, `batch_size` and `val_images`
# arrive from inputs=...; `losses`, `val_preds` and `device_name` are picked
# up from its namespace by outputs=[...].
TRAIN_CODE = """
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier(num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
losses = []
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    n_batches = 50
    for i in range(n_batches):
        images = torch.randn(batch_size, 3, 64, 64, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)

        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    avg = epoch_loss / n_batches
    losses.append(round(avg, 4))
    print(f"Epoch {epoch}/{epochs}  avg_loss={avg:.4f}")

# Classify the validation batch shipped from the client's namespace.
model.eval()
with torch.no_grad():
    val = torch.tensor(val_images, dtype=torch.float32, device=device)
    val_preds = model(val).argmax(dim=1).tolist()
device_name = str(device)
"""

OUTPUTS = ["losses", "val_preds", "device_name"]


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env")
        return

    # A small validation batch generated locally: 8 images, 3x32x32.
    # Plain nested lists — JSON-safe, well inside the 16 MB inline budget.
    val_images = [
        [[[random.random() for _ in range(32)] for _ in range(32)] for _ in range(3)]
        for _ in range(8)
    ]

    print("Submitting the training code block with local values...")
    handle = await client.run_code(
        TRAIN_CODE,
        inputs={"epochs": 3, "batch_size": 32, "val_images": val_images},
        outputs=OUTPUTS,
        pip=["torch"],
        timeout=300,
    )
    print(f"Task ID: {handle.task_id}")
    c = handle.classification
    print(f"Classification: tier={c.tier}, vram={c.min_vram_gb}GB, CU={c.compute_units}, method={c.analysis_method}")

    if client.estimate_only:
        return  # dry run — nothing to decode

    result = await handle.wait()
    values = decode_outputs(result.output, OUTPUTS)

    print(f"Losses per epoch: {values['losses']}")
    print(f"Validation predictions: {values['val_preds']}")
    print(f"Trained on: {values['device_name']}")
    print(f"Worker: {result.worker_id}  GPU: {result.actual_gpu}  Time: {result.execution_time_sec:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
