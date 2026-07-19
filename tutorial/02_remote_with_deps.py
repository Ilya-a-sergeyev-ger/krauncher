"""Tutorial 02: Remote task with pip dependencies.

Demonstrates installing packages at runtime inside the sandbox.
The `humanize` package is NOT in the base sandbox image —
the worker will `pip install` it before executing the function.
"""

import asyncio

from krauncher import KrauncherClient

client = KrauncherClient()


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
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    print("Submitting task with pip dependency: humanize...")
    handle = await format_big_numbers(value=1_234_567_890)
    print(f"Task submitted: {handle.task_id}")
    c = handle.classification
    print(f"Classification: {c.tier}, VRAM={c.min_vram_gb}GB, method={c.analysis_method}, confidence={c.confidence}")

    print("Waiting for result (pip install + execution)...")
    result = await handle
    print(f"Output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
