"""Tutorial 01: Simple remote task execution.
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
    cur = result.billing_currency
    print("── Billing ──────────────────────────────────")
    print(f"  Actual CU:      {result.actual_cu:.4f}")
    print(f"  Provider cost:  {result.provider_cost:.6f} {cur}")
    print(f"  Charged KU:     {result.charged_ku:.4f}")
    print("─────────────────────────────────────────────")

if __name__ == "__main__":
    asyncio.run(main())
