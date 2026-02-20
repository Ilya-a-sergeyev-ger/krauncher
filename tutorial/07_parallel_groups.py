"""Tutorial 07 — Parallel task groups (fan-out / fan-in).

Demonstrates submitting many tasks across several groups in parallel.
Each group routes to the same worker (host affinity), while different
groups may land on different workers.  After all tasks complete the
script prints a summary table showing which worker handled each group.

This pattern is useful for:
- Running independent experiments on separate machines
- Distributing hyperparameter sweeps across a GPU fleet
- Verifying that autoscaling provisions enough instances

Usage:
    CAS_API_KEY=cas_... python tutorial/07_parallel_groups.py

    # Custom sizes:
    CAS_API_KEY=cas_... python tutorial/07_parallel_groups.py --groups 4 --tasks-per-group 16
"""

import argparse
import asyncio
import os
import uuid
from collections import defaultdict

from krauncher import KrauncherClient

API_KEY = os.environ.get("CAS_API_KEY", "")
BROKER_URL = os.environ.get("CAS_BROKER_URL", "http://localhost:8000")

client = KrauncherClient(api_key=API_KEY, broker_url=BROKER_URL)


def compute(task_index, group_name):
    """Lightweight work — returns identity of the executing worker."""
    import platform
    import time

    time.sleep(0.5)  # simulate a bit of work
    return {
        "group": group_name,
        "task_index": task_index,
        "hostname": platform.node(),
        "pid": __import__("os").getpid(),
    }


async def main():
    parser = argparse.ArgumentParser(description="Parallel task groups demo")
    parser.add_argument("--groups", type=int, default=4, help="Number of groups")
    parser.add_argument(
        "--tasks-per-group", type=int, default=2, help="Tasks per group"
    )
    args = parser.parse_args()

    n_groups = args.groups
    n_tasks = args.tasks_per_group

    if not API_KEY:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    print(f"Submitting {n_groups} groups × {n_tasks} tasks = {n_groups * n_tasks} total\n")

    # ------------------------------------------------------------------
    # 1. Fan-out: submit all tasks across groups
    # ------------------------------------------------------------------
    group_ids = [f"group-{i}-{uuid.uuid4().hex[:6]}" for i in range(n_groups)]
    handles = []  # (group_index, task_index, handle)

    for gi, gid in enumerate(group_ids):
        # Apply decorator at runtime — compute is top-level so serialization works
        task_fn = client.task(vram_gb=1, timeout=120, group_id=gid)(compute)
        for ti in range(n_tasks):
            handle = await task_fn(task_index=ti, group_name=f"G{gi}")
            handles.append((gi, ti, handle))
            print(f"  submitted G{gi}/T{ti:02d}  id={handle.task_id}")

    print(f"\n{len(handles)} tasks submitted — waiting for results …\n")

    # ------------------------------------------------------------------
    # 2. Fan-in: gather all results concurrently
    # ------------------------------------------------------------------
    tasks = [h for _, _, h in handles]
    settled = await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # 3. Build summary
    # ------------------------------------------------------------------
    # group_index -> worker_id -> count
    group_workers: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors = []

    for (gi, ti, handle), outcome in zip(handles, settled):
        if isinstance(outcome, BaseException):
            errors.append((gi, ti, handle.task_id, str(outcome)))
            continue
        worker = outcome.worker_id or outcome.output.get("hostname", "?")
        group_workers[gi][worker] += 1

    # ------------------------------------------------------------------
    # 4. Print results table
    # ------------------------------------------------------------------
    print("=" * 68)
    print(f"{'Group':<8} {'Group ID':<22} {'Worker':<20} {'Tasks':>6}")
    print("-" * 68)

    for gi in range(n_groups):
        workers = group_workers.get(gi, {})
        first = True
        for worker, count in sorted(workers.items(), key=lambda x: -x[1]):
            group_label = f"G{gi}" if first else ""
            gid_label = group_ids[gi][:20] if first else ""
            print(f"{group_label:<8} {gid_label:<22} {worker:<20} {count:>6}")
            first = False

    print("=" * 68)

    # Affinity check
    print("\nHost-affinity verification:")
    all_ok = True
    for gi in range(n_groups):
        workers = group_workers.get(gi, {})
        unique = len(workers)
        tag = "OK" if unique == 1 else f"SPLIT across {unique} workers"
        if unique != 1:
            all_ok = False
        print(f"  G{gi}: {tag}")

    if all_ok:
        print(f"\nAll {n_groups} groups correctly pinned to their own worker.")
    else:
        print("\nWARNING: some groups were split — check fleet capacity.")

    # Unique workers used
    all_workers = set()
    for ws in group_workers.values():
        all_workers.update(ws.keys())
    print(f"\nTotal unique workers used: {len(all_workers)}")

    if errors:
        print(f"\n{len(errors)} task(s) failed:")
        for gi, ti, tid, err in errors:
            print(f"  G{gi}/T{ti:02d} [{tid}]: {err}")


if __name__ == "__main__":
    asyncio.run(main())
