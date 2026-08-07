# krauncher-mcp

An MCP server that gives an agent **one tool**: a pre-run cost estimate for a
GPU task, from static analysis of the code. The code is **never executed**.

It wraps the Krauncher *analyzer* (the assay), not the broker — there is no
dispatch, no execution, no market lineup. Just: how long will this cost, and
what does it need to run.

**The seconds are a relative signal for comparing code against code, not an
absolute forecast.** They are normalized to a fixed reference card (RTX PRO 6000
WS) so two estimates are comparable; the GPU and host your code actually runs on
will differ, so never read a second-count as the wall-clock you will get.
Compare variant-to-variant. This is an early **0.x** release — the model is
approximate and evolving.

## The `estimate` tool

Input: `code` — the task's Python source (a self-contained function; a
`@client.task`-decorated function is fine).

Output, on the reference card (**RTX PRO 6000 WS**, the card CU is normalized
to):

```jsonc
{
  "reference_card": "RTX PRO 6000 WS",
  "compute_sec": 20.2,      // the three phases of wall time on the ref card
  "setup_sec": 3.0,
  "io_sec": 2.1,
  "min_vram_gb": 6,         // raw requirement, no headroom margin
  "min_disk_gb": 10,
  "confidence": 1.0,        // 0-1
  "analysis_method": "ast", // "ast" | "llm"
  "cpu_only": false,
  "findings": [             // what the analyzer read from the code
    "num_epochs=1", "batch_size=16",
    "Recognized model: BERT Base (0.11B params)",
    "precision=fp16 from fp16=True"
  ]
}
```

The loop it is built for: **edit the run → estimate → keep what's cheaper →
repeat**, all before spending a GPU-second. The estimate is a static forecast,
not a guarantee; `confidence` and `analysis_method` say how much to trust it,
and a rough estimate never blocks — it returns a best effort.

What it does **not** return: the cost model's calibration coefficients or
weights. Only what the analyzer detected in the code leaves the server.

## Install

```sh
pip install -e .        # from this directory; also installs the analyzer client
```

The server needs an API key for the analyzer, in the environment:

```sh
export CAS_API_KEY=cas_...
```

Verify it works without wiring up a client — runs `estimate` on a sample task
and prints the contract:

```sh
krauncher-mcp --selftest
```

## Wire it into an MCP client

stdio transport; the console script is `krauncher-mcp`.

```jsonc
{
  "mcpServers": {
    "krauncher-analyzer": {
      "command": "krauncher-mcp",
      "env": { "CAS_API_KEY": "cas_..." }
    }
  }
}
```

## Scope

v1 is deliberately one tool. The per-GPU market lineup is intentionally left
out — the agent's job is to improve its code and know the cost before running,
and a pre-run estimate (even rough or partial) is the whole point.
