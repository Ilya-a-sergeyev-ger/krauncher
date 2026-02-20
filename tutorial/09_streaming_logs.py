"""Tutorial 09: Real-time log streaming via cas-relay.

Submits a training-loop task that prints per-epoch progress to stdout.
With on_log=, the client receives each line in real time via WebSocket
while the task is still executing — no need to wait for completion.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Start relay:   /opt/cas-relay/run.sh
    5. Install client: cd cas-client && pip install -e .
    6. Run: CAS_API_KEY=cas_... python tutorial/09_streaming_logs.py
"""

import asyncio
import os

from krauncher import KrauncherClient

client = KrauncherClient(
    api_key=os.environ.get("CAS_API_KEY", ""),
    broker_url=os.environ.get("CAS_BROKER_URL", "http://localhost:8000"),
)


@client.task(vram_gb=2, timeout=120)
def train_model(epochs: int, size: int):
    """Mini training loop — prints epoch stats to stdout as it runs."""
    import time
    import numpy as np

    np.random.seed(42)
    W = np.random.randn(size, size) * 0.01
    X = np.random.randn(size, size)
    y = np.random.randn(size, size)

    losses = []
    for epoch in range(1, epochs + 1):
        # Forward
        pred = X @ W
        loss = float(np.mean((pred - y) ** 2))

        # Backward (gradient step)
        grad = 2 * X.T @ (pred - y) / (size * size)
        W -= 0.01 * grad

        losses.append(loss)
        print(f"Epoch {epoch:3d}/{epochs}  loss={loss:.6f}", flush=True)
        time.sleep(0.3)

    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    return {
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "improvement_pct": round(improvement, 2),
        "epochs": epochs,
    }


def on_log(msg: dict) -> None:
    t = msg.get("type")
    d = msg.get("data", {})
    if t == "stdout":
        print(f"  {d.get('text', '').rstrip()}")
    elif t == "stderr":
        print(f"  [err] {d.get('text', '').rstrip()}")
    elif t == "event":
        name = d.get("name", "")
        if name not in ("stream_ended",):
            print(f"  [event] {name}")
    elif t == "metric":
        gpu = d.get("gpu_util_pct", 0)
        vram_used = d.get("vram_used_gb", 0)
        vram_total = d.get("vram_total_gb", 0)
        print(f"  [gpu]   {gpu:.0f}%  VRAM {vram_used:.1f}/{vram_total:.0f} GB")


async def main():
    api_key = os.environ.get("CAS_API_KEY")
    if not api_key:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    print("Submitting training task (10 epochs)...")
    handle = await train_model(epochs=10, size=256)
    print(f"Task ID: {handle.task_id}")
    print("─" * 50)

    result = await handle.wait(on_log=on_log)

    print("─" * 50)
    print(f"Initial loss:  {result.output['initial_loss']}")
    print(f"Final loss:    {result.output['final_loss']}")
    print(f"Improvement:   {result.output['improvement_pct']}%")
    print(f"Worker:        {result.worker_id}")
    print(f"GPU:           {result.actual_gpu}")
    print(f"Time:          {result.execution_time_sec:.2f}s")
    print(f"Cost:          ${result.cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
