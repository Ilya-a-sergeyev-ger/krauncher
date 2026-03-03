"""Tutorial 03: Error handling — remote traceback forwarding.

When a function raises an exception on the worker, the full traceback
is captured and forwarded to the client via TaskError.remote_traceback.
The exception message includes the remote traceback so it prints
naturally in the console, as if the error happened locally.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Install client: cd cas-client && pip install -e .
    5. Configure:     cas-client/.env (CAS_API_KEY, CAS_BROKER_URL, ...)
    6. Run: python tutorial/03_error_handling.py
"""

import asyncio

from krauncher import KrauncherClient, TaskError

client = KrauncherClient()


@client.task(vram_gb=1, timeout=60)
def buggy_function(x: int):
    result = 100 / x          # ZeroDivisionError when x=0
    data = {"value": result}
    return data["missing_key"] # KeyError (never reached when x=0)


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("=== Test 1: ZeroDivisionError ===\n")
    handle = await buggy_function(x=0)
    print(f"Task submitted: {handle.task_id}")
    c = handle.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB, method={c.analysis_method}, confidence={c.confidence}")

    try:
        result = await handle
        print(f"Unexpected success: {result.output}")
    except TaskError as e:
        print(f"Caught TaskError: {e}")
        print(f"\n--- Programmatic access ---")
        print(f"  task_id:          {e.task_id}")
        print(f"  has traceback:    {e.remote_traceback is not None}")

    print("\n\n=== Test 2: KeyError ===\n")
    handle2 = await buggy_function(x=5)
    print(f"Task submitted: {handle2.task_id}")

    try:
        result2 = await handle2
        print(f"Unexpected success: {result2.output}")
    except TaskError as e:
        print(f"Caught TaskError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
