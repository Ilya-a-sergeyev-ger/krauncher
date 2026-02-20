"""Tutorial 01: Simple remote task execution.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Install client: cd cas-client && pip install -e .
    5. Run: CAS_API_KEY=cas_... python tutorial/01_remote_simple.py
"""

import asyncio
import os

from krauncher import KrauncherClient

client = KrauncherClient(
    api_key=os.environ.get("CAS_API_KEY", ""),
    broker_url=os.environ.get("CAS_BROKER_URL", "http://localhost:8000"),
)


@client.task(vram_gb=1, timeout=120)
def multiply_matrices(size: int):
    import numpy as np
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    result = np.dot(a, b)
    return {"mean": float(result.mean()), "size": size}


async def main():
    api_key = os.environ.get("CAS_API_KEY")
    if not api_key:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    print("Submitting task...")
    handle = await multiply_matrices(size=1000)
    print(f"Task submitted: {handle.task_id}")

    print("Waiting for result...")
    result = await handle

    print(f"Output: {result.output}")
    print(f"Worker: {result.worker_id}")
    print(f"GPU: {result.actual_gpu}")
    print(f"Time: {result.execution_time_sec:.2f}s")
    print(f"Cost: ${result.cost_usd:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
