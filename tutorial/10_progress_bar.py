# Copyright (c) 2024-2026 Ilya Sergeev. Licensed under the MIT License.

"""Tutorial 10: Real-time progress bar via relay streaming.

Demonstrates \r-based in-place progress bars streamed through cas-relay.
Each batch update uses carriage return to overwrite the current line;
the client receives the raw \r/\n delimiters and renders them in the terminal.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Start relay:   /opt/cas-relay/run.sh
    5. Install client: cd cas-client && pip install -e .
    6. Configure:     cas-client/.env (CAS_API_KEY, CAS_BROKER_URL, ...)
    7. Run: python tutorial/10_progress_bar.py [multiplier]
       multiplier: 1x (default), 2x, 3x — scales matrix size and batch count
"""

import asyncio
import sys

from krauncher import KrauncherClient

client = KrauncherClient()

BAR_WIDTH = 25


@client.task(timeout=120)
def train_with_progress(epochs: int, n_batches: int, size: int):
    """Training loop with per-batch progress bar rendered via \\r."""
    import time
    import numpy as np

    np.random.seed(42)
    W = np.random.randn(size, size) * 0.01
    bar_width = 25

    losses = []
    for epoch in range(1, epochs + 1):
        batch_losses = []

        for batch in range(n_batches):
            X = np.random.randn(size, size)
            y = np.random.randn(size, size)

            pred = X @ W
            loss = float(np.mean((pred - y) ** 2))
            grad = 2 * X.T @ (pred - y) / (size * size)
            W -= 0.005 * grad
            batch_losses.append(loss)

            done = batch + 1
            filled = int(bar_width * done / n_batches)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\rEpoch {epoch:2d}/{epochs} [{bar}] {done:2d}/{n_batches}",
                end="",
                flush=True,
            )
            time.sleep(0.08)

        epoch_loss = sum(batch_losses) / len(batch_losses)
        losses.append(epoch_loss)
        bar = "█" * bar_width
        print(
            f"\rEpoch {epoch:2d}/{epochs} [{bar}] {n_batches}/{n_batches}"
            f"  loss={epoch_loss:.6f}"
        )

    return {
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "improvement_pct": round((losses[0] - losses[-1]) / losses[0] * 100, 2),
    }


def on_log(msg: dict) -> None:
    t = msg.get("type")
    d = msg.get("data", {})
    if t == "stdout":
        # Preserve \r and \n exactly — \r causes terminal overwrite, \n advances line
        text = d.get("text", "")
        print(text, end="", flush=True)
    elif t == "event":
        name = d.get("name", "")
        if name == "execution_started":
            print(f"[event] {name}")
        elif name == "execution_complete":
            print(f"\n[event] {name}")
    elif t == "metric":
        pass  # skip GPU metrics for cleaner progress bar display


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    # Parse optional multiplier: 1x (default), 2x, 3x, ...
    mul = 1
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower().rstrip("x")
        try:
            mul = max(1, int(arg))
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [multiplier, e.g. 2x]")
            return

    epochs = 5
    n_batches = 40 * mul
    size = 512 * mul
    print(f"Submitting training task ({epochs} epochs × {n_batches} batches, size={size}, {mul}x)...")
    handle = await train_with_progress(epochs=epochs, n_batches=n_batches, size=size)
    print(f"Task ID: {handle.task_id}")
    c = handle.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB, workload={c.workload_type}, method={c.analysis_method}, confidence={c.confidence}")
    print("─" * 55)

    result = await handle.wait(on_log=on_log)

    print("─" * 55)
    print(f"Initial loss:  {result.output['initial_loss']}")
    print(f"Final loss:    {result.output['final_loss']}")
    print(f"Improvement:   {result.output['improvement_pct']}%")
    print(f"Worker:        {result.worker_id}  GPU: {result.actual_gpu}")
    print(f"Time:          {result.execution_time_sec:.2f}s")
    print()
    cur = result.billing_currency
    print("── Billing ──────────────────────────────────")
    print(f"  Actual CU:      {result.actual_cu:.4f}")
    print(f"  Provider cost:  {result.provider_cost:.6f} {cur}")
    print(f"  Charged KU:     {result.charged_ku:.4f}")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    asyncio.run(main())
