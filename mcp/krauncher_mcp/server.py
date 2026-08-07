# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Krauncher analyzer MCP server.

Exposes ONE tool, `estimate`: a pre-execution cost profile for a GPU task,
from static analysis of the code (the code is never run). Wraps the client's
`estimate_code` path — the analyzer, not the broker. No dispatch, no execution,
no market lineup.

The reference card is the RTX PRO 6000 WS, the card CU is normalized to, so
`compute_units / 1000 == seconds on that card`; the three phase times fall
straight out of the CU breakdown.

Only what the analyzer *reads from the code* is exposed. The cost model's own
calibration coefficients and weights are never returned.
"""
from __future__ import annotations

import dataclasses
import logging

from mcp.server.mcpserver import MCPServer

from krauncher import KrauncherClient

from . import __version__

# Keep the closed core off the server's stderr. The analyzer logs its CU
# derivation — including calibration K-coefficients — at DEBUG, and the HTTP
# client logs the internal analyzer address; a host may surface a server's
# stderr. The tool *output* is already scrubbed; this covers the logs too.
for _noisy in ("krauncher", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Current CU anchor. If calibration re-anchors to a different SKU, change this
# one string — the whole contract is expressed on this card.
REFERENCE_CARD = "RTX PRO 6000 WS"

mcp = MCPServer("krauncher-analyzer", version=__version__)

_client: KrauncherClient | None = None


def _get_client() -> KrauncherClient:
    global _client
    if _client is None:
        # Reads CAS_API_KEY / KRAUNCHER_API_KEY (or a .env in CWD).
        _client = KrauncherClient()
    return _client


def _clean_findings(extra_debug: dict) -> list[str]:
    """The analyzer's read of the code — model, batch, precision, samples, etc.

    Never the cost model's calibration constants: we source from
    `extra_debug['findings']` (detection only, not the CU math in `cu_findings`)
    and defensively drop any line that still carries a K-coefficient.
    """
    out = []
    for line in (extra_debug or {}).get("findings", []) or []:
        low = line.lower()
        if "k_warmup" in low or "k_step" in low or "k_call" in low \
                or "k_byte" in low or "k_prefill" in low or "k_coeff" in low:
            continue
        out.append(line)
    return out


@mcp.tool()
async def estimate(code: str) -> dict:
    """Get a GPU task's cost profile *before* running it, from static analysis.

    Call this whenever you are about to run a GPU job — a training or inference
    task — and want to know the cost, or to compare variants of the run before
    spending any GPU time. The code is analyzed, never executed. Edit the run
    (card, precision, batch size, sequence length), re-estimate, keep the
    cheaper one, repeat — the whole loop costs no GPU-seconds.

    The seconds are a RELATIVE signal for comparing code against code, not an
    absolute forecast. They are normalized to a fixed reference card (RTX PRO
    6000 WS) so two estimates are comparable; the real GPU and host the code
    ends up running on will differ, so never read a second-count as the
    wall-clock you will actually get. Compare variant-to-variant, not to the
    clock. (Early 0.x release — the model is approximate and evolving.)

    Returns, on the reference card (RTX PRO 6000 WS):
      - compute_sec / setup_sec / io_sec: the three phases of wall time
      - min_vram_gb / min_disk_gb: what the task needs to run at all
      - confidence (0-1) and analysis_method ("ast" | "llm"): how much to trust it
      - cpu_only: true when the task makes no use of the GPU
      - findings: what the analyzer detected in the code (model, batch,
        precision, sample count, ...)

    Even a rough, low-confidence, or partial estimate is useful — it never
    blocks. On any failure it returns the same shape with an `error` string and
    confidence 0, never an exception.

    Args:
        code: the task's Python source — a self-contained function (a
            `@client.task`-decorated function is fine).
    """
    try:
        cls = await _get_client().estimate_code(code)
    except Exception as exc:  # best effort — hand the agent a usable shape
        return {
            "reference_card": REFERENCE_CARD,
            "compute_sec": None, "setup_sec": None, "io_sec": None,
            "min_vram_gb": None, "min_disk_gb": None,
            "confidence": 0.0, "analysis_method": None, "cpu_only": None,
            "findings": [],
            "error": f"estimate unavailable: {type(exc).__name__}: {exc}"[:300],
        }
    d = dataclasses.asdict(cls)

    cu_total = d.get("compute_units") or 0.0
    cu_io = d.get("cu_io") or 0.0
    cu_setup = d.get("cu_setup") or 0.0
    # ref-card seconds = CU / 1000; compute = total minus the io + setup phases.
    compute_sec = max(cu_total - cu_io - cu_setup, 0.0) / 1000.0

    extra = d.get("extra_debug") or {}
    # Both requirements come from top-level classification fields (krauncher
    # >=0.2.9), so they are present and consistent on every path — including the
    # safety-net fallback, where disk is the analyzer's floor rather than null.
    min_vram = d.get("min_vram_gb")
    min_disk = d.get("min_disk_gb")

    return {
        "reference_card": REFERENCE_CARD,
        "compute_sec": round(compute_sec, 1),
        "setup_sec": round(cu_setup / 1000.0, 1),
        "io_sec": round(cu_io / 1000.0, 1),
        "min_vram_gb": min_vram,
        "min_disk_gb": min_disk,
        "confidence": d.get("confidence"),
        "analysis_method": d.get("analysis_method"),
        "cpu_only": bool(d.get("cpu_only", False)),
        "findings": _clean_findings(extra),
    }


# A tiny task used by `--selftest` to exercise the contract without an MCP
# client (still hits the analyzer, so it needs CAS_API_KEY).
_SELFTEST_CODE = """def finetune(batch_size: int = 16):
    from transformers import (AutoModelForSequenceClassification,
                              TrainingArguments)
    model = AutoModelForSequenceClassification.from_pretrained(
        "/data/google-bert__bert-base-uncased", num_labels=2)
    args = TrainingArguments(
        output_dir="/tmp/o", per_device_train_batch_size=batch_size,
        num_train_epochs=1, fp16=True)
    return 1
"""


def _selftest() -> int:
    import asyncio
    import json

    out = asyncio.run(estimate(_SELFTEST_CODE))
    print(json.dumps(out, indent=2))
    ok = "error" not in out and out.get("compute_sec") is not None
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    """Console entry point — runs the server over stdio (or `--selftest`)."""
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    mcp.run()


if __name__ == "__main__":
    main()
