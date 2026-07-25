"""Tutorial 54: Files in, files out — text and binary.

Values return as JSON; files do not. This is how you hand a task a file and get
back what it produced — with nothing in between.

Both directions ride the same encrypted package that already carries your code
to the worker. Files you send are laid beside the task, in its working
directory; files the task writes beside itself come back with the result. No
storage, no mount paths to remember, and the broker sees none of it.

Files travel as raw bytes, never base64: a PNG made on the GPU arrives here
byte for byte. That is why the frame states each file's length instead of using
a separator — binary content contains every byte value, including whatever you
might have picked as one.

    Client                              Worker
    ──────                              ──────
    transform(files={"a.txt": …})   →   ./a.txt         laid beside the task
    result.download(dir)            ←   ./a.upper.txt   written by the task
                                    ←   ./gradient.png  made on the GPU host

Both directions share the 16 MB payload/result budget — enough for images,
documents, adapters. For a dataset or a checkpoint, use a volume (`volume=`)
instead; you get told if you overshoot rather than finding out on the wire.

Run it from a folder holding your .env (CAS_API_KEY):

    python tutorial/54_artifact_roundtrip.py
"""

import asyncio
from pathlib import Path

from krauncher import KrauncherClient

client = KrauncherClient()

LOCAL_OUT = Path("received")
SAMPLE = "sample.txt"
PNG_SIZE = (160, 120)


@client.task(vram_gb=1, timeout=300, artifacts=True)
def transform(width=160, height=120):
    """Transform the file that came along, and draw a PNG from scratch."""
    import struct
    import zlib
    from pathlib import Path

    written = []

    # ── text: read what was sent, write the transformed copy beside it ──
    for path in sorted(Path(".").glob("*.txt")):
        out = Path(f"{path.stem}.upper{path.suffix}")
        out.write_text(path.read_text().upper())
        written.append(out.name)
        print(f"{path.name} -> {out.name}", flush=True)

    # ── binary: a real PNG, encoded with the standard library only ──
    # Stands in for whatever the GPU actually renders; the point is that the
    # bytes below arrive on your machine unchanged.
    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body)))

    rows = b"".join(
        b"\x00" + bytes(  # 0 = no per-row filter
            v
            for x in range(width)
            for v in (255 * x // width, 255 * y // height, 128)
        )
        for y in range(height)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )
    Path("gradient.png").write_bytes(png)
    written.append("gradient.png")
    print(f"gradient.png -> {len(png)} bytes", flush=True)

    # The return value stays JSON; the files travel on their own.
    return {"written": written, "png_bytes": len(png)}


async def main():
    if not client.api_key:
        print("Set CAS_API_KEY in .env. New accounts get free credits.")
        return

    payload = b"the quick brown fox\njumps over the lazy dog\n"
    print(f"sending {SAMPLE} ({len(payload)} bytes) with the task")

    width, height = PNG_SIZE
    handle = await transform(files={SAMPLE: payload}, width=width, height=height)
    print(f"submitted {handle.task_id} — waiting...")
    result = await handle

    print(f"task returned: {result.output}")
    print(f"ran on {result.actual_gpu}, "
          f"{result.total_charged_ku} KU ({result.total_charged_local} "
          f"{result.billing_currency})")

    print(f"\nartifacts: {result.files}")
    count = result.download(str(LOCAL_OUT))
    print(f"wrote {count} file(s) into {LOCAL_OUT.resolve()}:")

    text = (LOCAL_OUT / f"{Path(SAMPLE).stem}.upper.txt").read_text()
    print(f"  text: {text!r}")

    png = (LOCAL_OUT / "gradient.png").read_bytes()
    intact = (
        png.startswith(b"\x89PNG\r\n\x1a\n")
        and png.endswith(b"IEND\xae\x42\x60\x82")
        and len(png) == result.output["png_bytes"]
    )
    print(f"  png : {len(png)} bytes, signature and length intact: {intact}")
    print(f"  open {LOCAL_OUT / 'gradient.png'} to see it")


if __name__ == "__main__":
    asyncio.run(main())
