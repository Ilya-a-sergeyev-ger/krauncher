"""Tutorial 04: Execution timeout — resource protection.

When a function hangs on the worker (e.g. ``while True: pass``), the
worker kills the container after the configured timeout and reports
status="timeout".  The client raises ``RemoteTimeout`` — a subclass
of ``TaskError`` — so you can catch it specifically or let a generic
``TaskError`` handler deal with it.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Install client: cd cas-client && pip install -e .
    5. Run: CAS_API_KEY=cas_... python tutorial/04_timeout.py
"""

import asyncio
import os

from krauncher import KrauncherClient, RemoteTimeout, TaskError

client = KrauncherClient(
    api_key=os.environ.get("CAS_API_KEY", ""),
    broker_url=os.environ.get("CAS_BROKER_URL", "http://localhost:8000"),
)


@client.task(vram_gb=1, timeout=10)
def hang_forever():
    while True:
        pass


async def main():
    api_key = os.environ.get("CAS_API_KEY")
    if not api_key:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    print("=== Timeout demo: while True: pass with timeout=10s ===\n")
    handle = await hang_forever()
    print(f"Task submitted: {handle.task_id}")
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
