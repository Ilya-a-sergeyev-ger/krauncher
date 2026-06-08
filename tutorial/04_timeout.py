"""Tutorial 04: Execution timeout — resource protection.

When a function hangs on the worker (e.g. ``while True: pass``), the
worker kills the container after the configured timeout and reports
status="timeout".  The client raises ``RemoteTimeout`` — a subclass
of ``TaskError`` — so you can catch it specifically or let a generic
``TaskError`` handler deal with it.
"""

import asyncio

from krauncher import KrauncherClient, RemoteTimeout, TaskError

client = KrauncherClient()


@client.task(vram_gb=1, timeout=10)
def hang_forever():
    while True:
        pass


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("=== Timeout demo: while True: pass with timeout=10s ===\n")
    handle = await hang_forever()
    print(f"Task submitted: {handle.task_id}")
    c = handle.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB, method={c.analysis_method}, confidence={c.confidence}")
    print("Waiting for worker to kill the container (~10s)...\n")

    try:
        result = await handle
        print(f"Unexpected success: {result.output}")
    except RemoteTimeout as e:
        print(f"Caught RemoteTimeout: {e}")
        print(f"\n--- Programmatic access ---")
        print(f"  task_id:     {e.task_id}")
        print(f"  timeout_sec: {e.timeout_sec}")
        print(f"  is TaskError: {isinstance(e, TaskError)}")
    except TaskError as e:
        print(f"Caught generic TaskError (unexpected): {e}")


if __name__ == "__main__":
    asyncio.run(main())
