# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Krauncher analyzer MCP server.

Exposes ONE tool, `estimate_gpu_time_and_cost`: a pre-execution time and cost
profile for a GPU task, from static analysis of the code (the code is never
run). The tool marks itself `anthropic/alwaysLoad`, so its schema is present
from the first turn instead of waiting behind a tool search: an agent asked to
make a GPU job faster answered from the source twice, both times reporting the
tool as deferred and never loaded. With a Krauncher API key it uses the keyed
client; without one it calls the public analyzer keyless
(subject to a per-IP daily quota). Either way it is the analyzer, not the
broker — no dispatch, no execution, no market lineup.

The reference card is the RTX PRO 6000 WS, the card CU is normalized to, so
`compute_units / 1000 == seconds on that card`; the three phase times fall
straight out of the CU breakdown.

Only what the analyzer *reads from the code* is exposed. The cost model's own
calibration coefficients and weights are never returned.
"""
from __future__ import annotations

import dataclasses
import logging
import os

from mcp.server.mcpserver import MCPServer

from krauncher import KrauncherClient
from krauncher.analyzer import AnalyzerClient

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

# The public analyzer for keyless (no-key) use. Overridable for self-hosting.
_DEFAULT_ANALYZER_URL = "https://analyzer.krauncher.com"

# Server instructions load at session start even when tool schemas are deferred
# (Claude Code's tool search is on by default: only tool names and this text
# reach the model on the first turn). So the "when to reach for this" argument
# has to live HERE, not only in the tool description — a host that defers
# schemas never shows the description until something already made it search.
# Truncated at 2KB by the client; keep the decisive sentences at the top.
_INSTRUCTIONS = """\
GPU job cost and runtime from static analysis of the code.

One tool, `estimate_gpu_time_and_cost`: give it a training or inference job's
Python source and it returns the time on a fixed reference card, the VRAM and
disk the job needs, the run parameters that move the time with the values found
in the code, and how far the same code scatters across real hosts. The numbers
come from a corpus of measured runs, not from reading the source.

The code is analyzed, never executed, so a call spends no GPU time and can be
repeated on variants of the same job.

What it answers: how long a GPU job will take, what is holding its performance
back, whether the GPU is actually utilized or left waiting on the host, and
which run parameters to change to make the code faster or cheaper.
"""

mcp = MCPServer("gpu-estimator", version=__version__, instructions=_INSTRUCTIONS)

_client: KrauncherClient | None = None
_keyless: AnalyzerClient | None = None


def _analyzer_url() -> str:
    return os.environ.get("KRAUNCHER_ANALYZER_URL", _DEFAULT_ANALYZER_URL).rstrip("/")


def _get_client() -> KrauncherClient | None:
    """The keyed client, or None when no API key is configured (env or .env)."""
    global _client
    if _client is None:
        try:
            # Reads KRAUNCHER_API_KEY / CAS_API_KEY (or a .env in CWD).
            _client = KrauncherClient()
        except Exception:
            return None
    return _client


def _get_keyless() -> AnalyzerClient:
    """Keyless client for the public analyzer — no key, quota'd per IP."""
    global _keyless
    if _keyless is None:
        ac = AnalyzerClient(analyzer_url=_analyzer_url(), token=None, timeout=30.0)
        # Tag the surface (telemetry + AST-only routing on the analyzer). The
        # client exposes no setter, so set the request header directly.
        ac._headers["X-Client-Surface"] = "mcp"
        _keyless = ac
    return _keyless


async def _classify(code: str):
    """Keyed path when a key is configured, else the keyless public analyzer.
    Both return a TaskClassification."""
    client = _get_client()
    if client is not None:
        return await client.estimate_code(code)
    return await _get_keyless().classify(code)


def _error(msg: str) -> dict:
    return {
        "reference_card": REFERENCE_CARD,
        "compute_sec": None, "setup_sec": None, "io_sec": None,
        "min_vram_gb": None, "min_disk_gb": None,
        "confidence": 0.0, "analysis_method": None, "cpu_only": None,
        "spread": None, "spread_reason": None, "calibration_basis": None,
        "knobs": [], "findings": [],
        "error": msg[:300],
    }


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


# Run parameters that move the time without changing what the code produces.
# These are ALWAYS reported, detected or not: a knob the analyzer could not see
# (value None) is itself the useful answer — the value is not a literal in the
# source, so nothing can price it until it is one.
_SAME_WORK_KNOBS = (
    "num_workers",
    "batch_size",
    "gradient_accumulation_steps",
)

# Parameters that change the SIZE of the job — fewer epochs, fewer steps, a
# shorter sequence. Reported only when found, because a lower number here buys a
# smaller result rather than a cheaper one. Listing them as absent would invite
# an agent to "save" by cutting the work.
_WORK_KNOBS = (
    "num_epochs",
    "epochs",
    "num_train_epochs",
    "max_steps",
    "seq_len",
    "image_size",
)

# DataLoader worker counts arrive under whatever spelling the code used —
# num_workers, dataloader_num_workers, n_workers, workers, nw. Mirrors the
# analyzer's own rule so every spelling lands on one canonical knob name.
_WORKER_ABBREV = frozenset({"nw", "n_w", "num_w"})


def _is_worker_name(name: str) -> bool:
    n = name.lower()
    return n.endswith("workers") or n.startswith("num_wo") or n in _WORKER_ABBREV


def _knobs(findings: list[str]) -> list[dict]:
    """The run parameters to re-estimate, with the values found in the code.

    A finding reads `name=value`, sometimes with a trailing note
    (`image_size=224 from dataset default`). Names that are not knobs — the
    model, the dataset, the optimizer — stay in `findings` and out of here.

    Every same-work knob is listed whether or not it was found; `value: None`
    means the analyzer did not see it as a literal in this code (it arrives as a
    call argument, from a config, or from the environment), so it cannot be
    priced until the code names it.
    """
    found: dict[str, str] = {}
    for line in findings:
        raw, sep, rest = line.partition("=")
        raw, rest = raw.strip(), rest.strip()
        if not sep or not rest or " " in raw:
            continue
        name = "num_workers" if _is_worker_name(raw) else raw
        found.setdefault(name, rest.split()[0])

    out = [
        {"name": name, "value": found.get(name), "same_work": True}
        for name in _SAME_WORK_KNOBS
    ]
    out += [
        {"name": name, "value": found[name], "same_work": False}
        for name in _WORK_KNOBS if name in found
    ]
    return out


@mcp.tool(meta={"anthropic/alwaysLoad": True})
async def estimate_gpu_time_and_cost(code: str) -> dict:
    """Runtime, cost and performance analysis of a GPU job, before running it.

    Use it to predict how long a training or inference job will take, find what
    holds its performance back, see whether the GPU is utilized or left waiting
    on the host, and pick the run parameters that make the code faster or
    cheaper. Numbers come from a corpus of measured runs — a
    runtime reasoned out of source is routinely wrong by several times. The code
    is analyzed, never executed: change a parameter, re-estimate, keep the
    cheaper variant — zero GPU-seconds.

    Returns, on a fixed reference card (RTX PRO 6000 WS): compute_sec /
    setup_sec / io_sec, min_vram_gb / min_disk_gb, cpu_only, confidence (0-1),
    analysis_method ("ast" | "llm"), findings (what the analyzer read from the
    code), plus knobs / spread / spread_reason / calibration_basis:

    `knobs` is the shortlist worth re-estimating — change their VALUES ONLY, not
    the model, architecture, dataset or procedure. `value: null` means it is not
    a literal in this source (it arrives as an argument, a config entry or an
    env var), so name it in the code and estimate again. `same_work: false`
    (epochs, steps, sequence length) shrinks the job itself: report that as a
    change of task, never as a saving.

    `spread` is how far the same code scatters across real hosts: 1.5 means the
    slow end of the measured population runs about 1.5x the estimate. It is a
    measured factor, not doubt about the reading — confidence 1.0 with a large
    spread says the GPU is not what governs this job's time, the card waits on
    the host. The lever is GPU utilization: find what leaves the card idle (data
    loaded in the main process, per-item preprocessing, synchronous transfers, a
    batch too small to fill it). `spread_reason` names it.

    `calibration_basis` is "calibrated", "extrapolated" (a neighbour answered)
    or "uncalibrated" (read the seconds as an order of magnitude).

    The seconds compare code against code on a fixed card, never the wall-clock
    you will get. On failure the same shape returns with an `error` and
    confidence 0.
    """
    try:
        cls = await _classify(code)
    except Exception as exc:  # best effort — hand the agent a usable shape
        resp = getattr(exc, "response", None)
        if getattr(resp, "status_code", None) == 429:
            # Keyless daily quota hit — surface the registration CTA verbatim.
            try:
                detail = (resp.json() or {}).get("detail")
            except Exception:
                detail = None
            return _error(detail or (
                "Anonymous daily limit reached — register free at "
                "https://krauncher.com/?ref=mcp for a larger quota."))
        return _error(f"estimate unavailable: {type(exc).__name__}: {exc}")
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
    findings = _clean_findings(extra)

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
        # Analyzer-side honesty fields, carried through the client's untyped
        # duration_estimate pass-through. Absent on older analyzers → null.
        "spread": extra.get("spread"),
        "spread_reason": extra.get("spread_reason"),
        "calibration_basis": extra.get("calibration_basis"),
        "knobs": _knobs(findings),
        "findings": findings,
    }


# A tiny task used by `--selftest` to exercise the contract without an MCP
# client. Works with or without a key: with KRAUNCHER_API_KEY it uses the keyed path,
# otherwise the keyless public analyzer (subject to the per-IP quota).
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

    out = asyncio.run(estimate_gpu_time_and_cost(_SELFTEST_CODE))
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
