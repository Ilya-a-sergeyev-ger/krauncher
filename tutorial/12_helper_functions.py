"""Tutorial 12: Helper functions — automatic multi-function serialization.

Demonstrates that krauncher automatically detects and includes helper
functions defined in the same module when they are called by a task.

You do NOT need to put all your code inside a single function — define
helpers alongside your task and they will be serialized together.

The helpers are collected via AST analysis (call graph walk) and sent
to the worker as a single code_string.  Transitive dependencies are
resolved automatically: if helper A calls helper B, both are included.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Install client: cd cas-client && pip install -e .
    5. Run: CAS_API_KEY=cas_... python tutorial/12_helper_functions.py
"""

import asyncio
import os

from krauncher import KrauncherClient

client = KrauncherClient(
    api_key=os.environ.get("CAS_API_KEY", ""),
    broker_url=os.environ.get("CAS_BROKER_URL", "http://localhost:8000"),
)


# ── Helper functions (same module, NOT inside the task) ─────────────


def normalize(values: list[float]) -> list[float]:
    """Scale values to [0, 1] range."""
    lo, hi = min(values), max(values)
    rng = hi - lo or 1.0
    return [(v - lo) / rng for v in values]


def moving_average(values: list[float], window: int = 3) -> list[float]:
    """Simple moving average."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def preprocess(raw_data: list[float], window: int = 3) -> list[float]:
    """Normalize, then smooth — calls two other helpers."""
    normed = normalize(raw_data)
    return moving_average(normed, window)


# ── Task entry point ────────────────────────────────────────────────


@client.task(vram_gb=1, timeout=120)
def analyze(raw_data: list[float], window: int = 3):
    """Entry point that uses helpers from the same module."""
    smoothed = preprocess(raw_data, window)
    return {
        "input_len": len(raw_data),
        "smoothed_len": len(smoothed),
        "smoothed_min": round(min(smoothed), 4),
        "smoothed_max": round(max(smoothed), 4),
        "smoothed_mean": round(sum(smoothed) / len(smoothed), 4),
    }


# ── Main ────────────────────────────────────────────────────────────


async def main():
    api_key = os.environ.get("CAS_API_KEY")
    if not api_key:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    raw = [10.0, 25.0, 13.0, 42.0, 8.0, 37.0, 19.0, 50.0, 5.0, 30.0]

    print(f"Input data: {raw}")
    print(f"Submitting task with {3} helper functions...")
    handle = await analyze(raw_data=raw, window=4)
    print(f"Task submitted: {handle.task_id}")

    print("Waiting for result...")
    result = await handle

    print(f"\nOutput: {result.output}")
    print(f"GPU:    {result.actual_gpu}")
    print(f"Time:   {result.execution_time_sec:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
