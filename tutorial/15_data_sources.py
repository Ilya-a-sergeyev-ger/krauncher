"""Tutorial 15: Data Sources — named data with input/output workflow.

Demonstrates the named data sources feature: instead of passing raw URLs
in every task, you register data sources once (via Web UI or API) and
reference them by name.

Scenario:
    1. "source-data" — a registered data source pointing to a CSV file
       (registered in Web UI → Data Sources, with URLs and size_gb)
    2. "upload-folder" — a registered output data source (is_output=True)
       where the task uploads its results

The task reads the input CSV from /data, processes it, and writes
the result to /output — which syncs back to the output data source.

Data flow:
    Client                     Broker                      Worker
    ──────                     ──────                      ──────
    @task(data="source-data",  resolve "source-data"       Phase 2: download URLs → /data
          output="upload-folder")  → inject URLs,creds     Phase 3: execute user code
                               resolve "upload-folder"     Phase 3.5: sync /output → S3
                                   → inject s3_prefix

Prerequisites:
    1. Register two data sources in Web UI (http://localhost:5173/data-sources):

       a) "source-data"
          - URLs: https://krauncher.com/assets/tickers.csv
          - Size: 0.01 GB
          - is_output: OFF

       b) "upload-folder"
          - URLs: (empty or your S3 bucket URL)
          - Size: 1 GB
          - is_output: ON

    2. Seed API key:   python seed_api_key.py
    3. Start broker:   cd cas-broker && PYTHONPATH=src uvicorn broker.main:app --port 8000
    4. Start worker:   cd cas-worker && PYTHONPATH=src python -m worker.main
    5. Install client: cd cas-client && pip install -e .
    6. Configure:      cas-client/.env (CAS_API_KEY, CAS_BROKER_URL, ...)
    7. Run: python tutorial/15_data_sources.py
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()

# Data source names — must match what's registered in the Web UI
INPUT_SOURCE = "source-data"
OUTPUT_SOURCE = "upload-folder"


@client.task(
    vram_gb=1,
    timeout=120,
    data=INPUT_SOURCE,          # broker resolves → downloads to /data
    output=OUTPUT_SOURCE,       # broker resolves → syncs /output back to S3
)
def process_tickers():
    """Read tickers from /data, compute summary, save result to /output."""
    import csv
    import json
    import os

    # ── Step 1: List what was synced into /data ──
    data_files = os.listdir("/data")
    print(f"Files in /data: {data_files}")

    # ── Step 2: Read input CSV ──
    csv_path = "/data/tickers.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    # ── Step 3: Process — compute per-sector summary ──
    sectors = {}
    for row in rows:
        s = row["sector"]
        sectors[s] = sectors.get(s, 0) + 1

    top_sectors = dict(
        sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    summary = {
        "total_tickers": len(rows),
        "sector_count": len(sectors),
        "top_sectors": top_sectors,
    }

    # ── Step 4: Write result to /output ──
    os.makedirs("/output", exist_ok=True)
    output_path = "/output/sector_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {output_path}")

    # Also write a CSV version
    csv_output = "/output/sector_counts.csv"
    with open(csv_output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sector", "ticker_count"])
        for sector, count in top_sectors.items():
            w.writerow([sector, count])
    print(f"Wrote CSV to {csv_output}")

    output_files = os.listdir("/output")
    print(f"Files in /output: {output_files}")

    return summary


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print(f"Input data source:  '{INPUT_SOURCE}'")
    print(f"Output data source: '{OUTPUT_SOURCE}'")
    print()
    print("Submitting task...")
    handle = await process_tickers()
    print(f"Task submitted: {handle.task_id}")

    c = handle.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB")

    print("Waiting for result (download → process → upload)...")
    result = await handle

    output = result.output
    print(f"\nResults:")
    print(f"  Total tickers: {output['total_tickers']}")
    print(f"  Sectors found: {output['sector_count']}")
    print(f"\n  Top sectors:")
    for sector, count in output["top_sectors"].items():
        print(f"    {sector}: {count}")

    # Timing breakdown
    exec_sec = result.execution_time_sec - result.download_sec - result.pip_install_sec
    print(f"\nTiming:")
    print(f"  Queue wait: {result.queue_wait_sec:.2f}s")
    print(f"  Download:   {result.download_sec:.2f}s")
    print(f"  Execution:  {exec_sec:.2f}s")
    print(f"  Total:      {result.execution_time_sec:.2f}s")
    print(f"  Cost:       ${result.cost_usd:.6f}")

    print("\nOutput files were synced to the output data source.")
    print("Check your S3 bucket or the Web UI to see sector_summary.json and sector_counts.csv.")


if __name__ == "__main__":
    asyncio.run(main())
