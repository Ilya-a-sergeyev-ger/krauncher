# Copyright (c) 2024-2026 Ilya Sergeev. Licensed under the MIT License.

"""Tutorial 14: Custom PyTorch model — tests analyzer LLM path.

Uses a hand-written model (not in HF registry) so AST confidence stays low
and the analyzer falls through to LLM analysis.
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()


@client.task(timeout=300, pip=["torch"])
def train_custom_model(epochs: int, batch_size: int):
    """Train a custom CNN on synthetic image data."""
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
            print(f"\rEpoch {epoch}/{epochs}  batch {i+1}/{n_batches}  loss={loss.item():.4f}", end="")

        avg = epoch_loss / n_batches
        losses.append(avg)
        print(f"\rEpoch {epoch}/{epochs}  avg_loss={avg:.4f}          ")

    return {
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "device": str(device),
    }


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env")
        return

    print("Submitting custom model training task...")
    handle = await train_custom_model(epochs=5, batch_size=32)
    print(f"Task ID: {handle.task_id}")
    c = handle.classification
    print(f"Classification: tier={c.tier}, vram={c.min_vram_gb}GB, CU={c.compute_units}, method={c.analysis_method}, confidence={c.confidence}")

    result = await handle
    print(f"Result: {result.output}")
    print(f"Worker: {result.worker_id}  GPU: {result.actual_gpu}  Time: {result.execution_time_sec:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
