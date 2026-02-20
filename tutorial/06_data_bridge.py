"""Tutorial 06: Data Bridge — loading external data into the sandbox.

Demonstrates data_urls parameter: the worker downloads files from HTTP/S3
URLs *before* container startup and mounts them at /data (read-only).

The function reads a CSV dataset from /data/, computes statistics,
and returns results that depend on the downloaded data.

Data flow:
    1. Client sends data_urls=["https://..."] in the task request
    2. Worker's DataBridge downloads the files to host filesystem
    3. Downloaded directory is mounted into the container at /data (ro)
    4. User code reads from /data/<filename>

Prerequisites:
    1. Seed API key:  python seed_api_key.py
    2. Start broker:  cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    3. Start worker:  cd cas-worker && PYTHONPATH=src python -m worker.main
    4. Install client: cd cas-client && pip install -e .
    5. Run: CAS_API_KEY=cas_... python tutorial/06_data_bridge.py
"""

import asyncio
import os

from krauncher import KrauncherClient

DATA_URL = "https://krauncher.com/assets/tickers.csv"

client = KrauncherClient(
    api_key=os.environ.get("CAS_API_KEY", ""),
    broker_url=os.environ.get("CAS_BROKER_URL", "http://localhost:8000"),
)


@client.task(vram_gb=1, timeout=120, data_urls=[DATA_URL])
def analyze_tickers():
    """Read tickers.csv from /data, compute per-sector statistics."""
    import csv
    import os

    # List files available in /data
    data_files = os.listdir("/data")

    # Read CSV: ticker,sector,industry,market_cap
    rows = []
    csv_path = "/data/tickers.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Count tickers per sector
    sector_counts = {}
    for row in rows:
        sector = row["sector"]
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Count tickers per market_cap category
    cap_counts = {}
    for row in rows:
        cap = row["market_cap"]
        cap_counts[cap] = cap_counts.get(cap, 0) + 1

    # Top 5 sectors by ticker count
    top_sectors = dict(
        sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    )

    return {
        "data_files": data_files,
        "header": list(header),
        "total_rows": len(rows),
        "sector_count": len(sector_counts),
        "cap_categories": cap_counts,
        "top_sectors": top_sectors,
    }


async def main():
    api_key = os.environ.get("CAS_API_KEY")
    if not api_key:
        print("ERROR: Set CAS_API_KEY env var (run seed_api_key.py first)")
        return

    print(f"Submitting task with data_urls=[{DATA_URL}]")
    handle = await analyze_tickers()
    print(f"Task submitted: {handle.task_id}")

    print("Waiting for result (download + execution)...")
    result = await handle

    output = result.output
    print(f"\nFiles in /data: {output['data_files']}")
    print(f"CSV header: {output['header']}")
    print(f"Total rows: {output['total_rows']}")
    print(f"Sectors: {output['sector_count']}")
    print(f"\nMarket cap distribution:")
    for cap, count in output["cap_categories"].items():
        print(f"  {cap}: {count}")
    print(f"\nTop 5 sectors by ticker count:")
    for sector, count in output["top_sectors"].items():
        print(f"  {sector}: {count}")

    # Timing breakdown
    exec_sec = result.execution_time_sec - result.download_sec - result.pip_install_sec
    print(f"\nTiming Breakdown:")
    print(f"  Queue wait:   {result.queue_wait_sec:.2f}s")
    print(f"  Download:     {result.download_sec:.2f}s")
    print(f"  Execution:    {exec_sec:.2f}s")
    print(f"  Total:        {result.execution_time_sec:.2f}s")
    print(f"Cost: ${result.cost_usd:.6f}")

    # Verify data was actually loaded
    assert output["total_rows"] > 1000, f"Expected >1000 rows, got {output['total_rows']}"
    assert output["sector_count"] > 5, f"Expected >5 sectors, got {output['sector_count']}"
    print("\nAll assertions passed!")


if __name__ == "__main__":
    asyncio.run(main())
