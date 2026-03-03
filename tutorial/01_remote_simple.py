"""Tutorial 01: Simple remote task execution.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Install client: cd cas-client && pip install -e .
    5. Configure:     cas-client/.env (CAS_API_KEY, CAS_BROKER_URL, ...)
    6. Run: python tutorial/01_remote_simple.py
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()


@client.task(vram_gb=1, timeout=120)
def multiply_matrices(size: int):
    import numpy as np
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    result = np.dot(a, b)
    return {"mean": float(result.mean()), "size": size}


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting task...")
    handle = await multiply_matrices(size=1000)
    print(f"Task submitted: {handle.task_id}")
    c = handle.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB, method={c.analysis_method}, confidence={c.confidence}")

    print("Waiting for result...")
    result = await handle

    print(f"Output: {result.output}")
    print(f"Worker: {result.worker_id}")
    print(f"GPU:    {result.actual_gpu}")
    print(f"Time:   {result.execution_time_sec:.2f}s")
    print()
    print("── Billing ──────────────────────────────────")
    print(f"  Provider cost:  ${result.cost_usd:.6f} USD")
    cur = result.billing_currency
    if result.client_cost:
        print(f"  Net charge:     {result.client_cost:.6f} {cur}")
        if result.vat_rate_pct:
            print(f"  VAT ({result.vat_rate_pct:.1f}%):     {result.vat_amount:.6f} {cur}")
        print(f"  Total:          {result.total_cost:.6f} {cur}")
    else:
        print(f"  (billing info not yet available)")
    print("─────────────────────────────────────────────")

if __name__ == "__main__":
    asyncio.run(main())
