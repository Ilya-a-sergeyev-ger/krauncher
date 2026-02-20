"""Tutorial 02: Remote task with pip dependencies.

Demonstrates installing packages at runtime inside the sandbox.
The `humanize` package is NOT in the base sandbox image —
the worker will `pip install` it before executing the function.

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Install client: cd cas-client && pip install -e .
    5. Run: CAS_API_KEY=cas_... python tutorial/02_remote_with_deps.py
"""

import asyncio
import os

from krauncher import KrauncherClient

client = KrauncherClient(
    api_key=os.environ.get("CAS_API_KEY", ""),
    broker_url=os.environ.get("CAS_BROKER_URL", "http://localhost:8000"),
)


@client.task(vram_gb=1, timeout=120, pip=["humanize"])
def format_big_numbers(value: int):
    import humanize
    return {
        "original": value,
        "intword": humanize.intword(value),
        "intcomma": humanize.intcomma(value),
        "scientific": humanize.scientific(value),
    }


async def main():
    api_key = os.environ.get("CAS_API_KEY")
    if not api_key:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    print("Submitting task with pip dependency: humanize...")
    handle = await format_big_numbers(value=1_234_567_890)
    print(f"Task submitted: {handle.task_id}")

    print("Waiting for result (pip install + execution)...")
    result = await handle
    print(f"Output: {result.output}")

    # Timing breakdown
    print(f"\nTiming Breakdown:")
    print(f"  Queue wait:   {result.queue_wait_sec:.2f}s")
    print(f"  Pip install:  {result.pip_install_sec:.2f}s")
    exec_sec = result.execution_time_sec - result.pip_install_sec - result.download_sec
    print(f"  Execution:    {exec_sec:.2f}s")
    print(f"  Total:        {result.execution_time_sec:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
