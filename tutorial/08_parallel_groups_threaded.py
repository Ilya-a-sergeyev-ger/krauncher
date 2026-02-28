"""Tutorial 08 — Parallel task groups with threaded submission.

Same as tutorial 07 but each group submits AND waits for its tasks
from a dedicated thread, so all groups fire concurrently.  This
stresses the broker's group routing under real concurrent load.

Usage:
    CAS_API_KEY=cas_... python tutorial/08_parallel_groups_threaded.py

    # Custom sizes:
    CAS_API_KEY=cas_... python tutorial/08_parallel_groups_threaded.py --groups 4 --tasks-per-group 4
"""

import argparse
import asyncio
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from krauncher import KrauncherClient

API_KEY = os.environ.get("CAS_API_KEY", "")
BROKER_URL = os.environ.get("CAS_BROKER_URL", "http://localhost:8000")


def compute(task_index, group_name):
    """Lightweight work — returns identity of the executing worker."""
    import platform
    import time

    time.sleep(0.5)
    return {
        "group": group_name,
        "task_index": task_index,
        "hostname": platform.node(),
        "pid": __import__("os").getpid(),
    }


def _run_group(gi: int, group_id: str, n_tasks: int) -> list[tuple[int, int, str, object]]:
    """Submit and wait for all tasks in one group (runs in its own thread).

    Returns list of (group_index, task_index, task_id, result_or_error).
    Each successful result is a TaskResult object.
    """
    loop = asyncio.new_event_loop()
    client = KrauncherClient(api_key=API_KEY, broker_url=BROKER_URL)
    task_fn = client.task(vram_gb=1, timeout=120, group_id=group_id)(compute)

    async def _submit_and_wait():
        handles = []
        for ti in range(n_tasks):
            handle = await task_fn(task_index=ti, group_name=f"G{gi}")
            handles.append((ti, handle))
            print(f"  submitted G{gi}/T{ti:02d}  id={handle.task_id}")

        results = []
        settled = await asyncio.gather(*[h for _, h in handles], return_exceptions=True)
        for (ti, handle), outcome in zip(handles, settled):
            results.append((gi, ti, str(handle.task_id), outcome))
        return results

    try:
        return loop.run_until_complete(_submit_and_wait())
    finally:
        loop.close()


def main():
    parser = argparse.ArgumentParser(description="Parallel task groups demo (threaded)")
    parser.add_argument("--groups", type=int, default=4, help="Number of groups")
    parser.add_argument(
        "--tasks-per-group", type=int, default=8, help="Tasks per group"
    )
    args = parser.parse_args()

    n_groups = args.groups
    n_tasks = args.tasks_per_group

    if not API_KEY:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    print(f"Submitting {n_groups} groups × {n_tasks} tasks = {n_groups * n_tasks} total")
    print(f"Each group submits from its own thread\n")

    # ------------------------------------------------------------------
    # 1. Fan-out: each thread submits + waits for its group
    # ------------------------------------------------------------------
    group_ids = [f"group-{i}-{uuid.uuid4().hex[:6]}" for i in range(n_groups)]
    all_results = []

    with ThreadPoolExecutor(max_workers=n_groups) as pool:
        futures = {
            pool.submit(_run_group, gi, gid, n_tasks): gi
            for gi, gid in enumerate(group_ids)
        }
        for future in as_completed(futures):
            all_results.extend(future.result())

    all_results.sort(key=lambda x: (x[0], x[1]))

    # ------------------------------------------------------------------
    # 2. Build summary
    # ------------------------------------------------------------------
    group_workers: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors = []

    for gi, ti, task_id, outcome in all_results:
        if isinstance(outcome, BaseException):
            errors.append((gi, ti, task_id, str(outcome)))
            continue
        worker = outcome.worker_id or outcome.output.get("hostname", "?")
        group_workers[gi][worker] += 1

    # ------------------------------------------------------------------
    # 3. Print results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
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

    # ------------------------------------------------------------------
    # 4. Billing summary
    # ------------------------------------------------------------------
    successful = [
        outcome for _, _, _, outcome in all_results
        if not isinstance(outcome, BaseException)
    ]
    if successful:
        cur            = successful[0].billing_currency or "USD"
        vat_rate       = successful[0].vat_rate_pct
        total_provider = sum(r.cost_usd for r in successful)
        total_client   = round(sum(r.client_cost for r in successful), 6)
        total_charge   = round(sum(r.total_cost  for r in successful), 6)
        # Derive total VAT from actual charged amounts — individual per-task
        # vat_amount fields may round to 0 for micro-transactions.
        total_vat      = round(total_charge - total_client, 6)

        print("\n── Billing ──────────────────────────────────")
        print(f"  Tasks completed:  {len(successful)}")
        print(f"  Provider cost:    ${total_provider:.6f} USD")
        print(f"  Net charge:       {total_client:.6f} {cur}")
        if vat_rate and total_vat > 0:
            print(f"  VAT ({vat_rate:.1f}%):       {total_vat:.6f} {cur}")
        elif vat_rate:
            print(f"  VAT ({vat_rate:.1f}%):       0 (below per-task rounding threshold)")
        print(f"  Total charged:    {total_charge:.6f} {cur}")
        print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()
