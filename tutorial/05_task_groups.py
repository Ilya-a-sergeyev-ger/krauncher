"""Tutorial 05 — Task Groups (host affinity).

Two tasks with the same group_id are guaranteed to run on the same worker.
This is useful when tasks share data cached on the host, use a common
model loaded in GPU memory, or otherwise benefit from co-location.

Usage:
    CAS_API_KEY=cas_... python tutorial/05_task_groups.py
"""

import asyncio
import os
import uuid

from krauncher import KrauncherClient

API_KEY = os.environ.get("CAS_API_KEY", "")
BROKER_URL = os.environ.get("CAS_BROKER_URL", "http://localhost:8000")

client = KrauncherClient(api_key=API_KEY, broker_url=BROKER_URL)

# All tasks in this group will be routed to the same worker
GROUP = f"experiment-{uuid.uuid4().hex[:8]}"


@client.task(vram_gb=1, timeout=60, group_id=GROUP)
def step_one(value):
    """First step — runs on an assigned worker."""
    import platform

    return {
        "step": 1,
        "value": value * 2,
        "hostname": platform.node(),
    }


@client.task(vram_gb=1, timeout=60, group_id=GROUP)
def step_two(value):
    """Second step — guaranteed to run on the same worker as step_one."""
    import platform

    return {
        "step": 2,
        "value": value + 100,
        "hostname": platform.node(),
    }


async def main():
    print(f"Task group: {GROUP}\n")

    # Submit both tasks
    handle1 = await step_one(value=21)
    handle2 = await step_two(value=42)
    print(f"Submitted: {handle1.task_id}, {handle2.task_id}")

    # Wait for results
    result1, result2 = await asyncio.gather(handle1, handle2)

    print(f"\nStep 1: value={result1.output['value']}, worker={result1.worker_id}")
    print(f"Step 2: value={result2.output['value']}, worker={result2.worker_id}")

    # Verify same worker
    if result1.worker_id == result2.worker_id:
        print(f"\nHost affinity confirmed: both tasks ran on {result1.worker_id}")
    else:
        print(f"\nWARNING: tasks ran on different workers!")
        print(f"  step_one  → {result1.worker_id}")
        print(f"  step_two  → {result2.worker_id}")


if __name__ == "__main__":
    asyncio.run(main())
