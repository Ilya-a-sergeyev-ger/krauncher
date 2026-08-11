# krauncher-mcp

<!-- mcp-name: io.github.Ilya-a-sergeyev-ger/krauncher-mcp -->

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
  "knobs": [                // the run parameters worth re-estimating
    { "name": "num_workers", "value": null, "same_work": true },
    { "name": "batch_size",  "value": "16", "same_work": true },
    { "name": "num_epochs",  "value": "1",  "same_work": false }
  ],
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

`knobs` is what turns that loop from guesswork into a shortlist. The parameters
that move the time without changing what the code produces are listed every
time, found or not; only the values are meant to change, since restructuring the
job (a smaller model, a different architecture, less data) is not what the number
is for.

- `value: null` — the analyzer did not see that parameter in the source. It
  arrives as a call argument, from a config, or from the environment, and
  nothing can price it until the code states it as a literal.
- `same_work: false` — epochs, steps, sequence length. These shrink the job
  itself, so a lower number is a different task rather than a cheaper one. They
  appear only when the code actually sets them.

What it does **not** return: the cost model's calibration coefficients or
weights. Only what the analyzer detected in the code leaves the server.

## Install

```sh
pip install -e .        # from this directory; also installs the analyzer client
```

An API key is **optional**. Without one the server calls the public analyzer
**keyless**, under a per-IP daily quota (when the quota is reached, `estimate`
returns a short note to register for a larger one). Set a key to use your own
account and skip the quota:

```sh
export KRAUNCHER_API_KEY=cas_...   # optional
```

Verify it works without wiring up a client — runs `estimate` on a sample task
and prints the contract:

```sh
krauncher-mcp --selftest
```

## Wire it into an MCP client

stdio transport; the console script is `krauncher-mcp`. No key needed — this runs
keyless against the public analyzer:

```jsonc
{
  "mcpServers": {
    "krauncher-analyzer": {
      "command": "krauncher-mcp"
    }
  }
}
```

To use your own account (keyed, exempt from the per-IP quota), add the key:

```jsonc
{
  "mcpServers": {
    "krauncher-analyzer": {
      "command": "krauncher-mcp",
      "env": { "KRAUNCHER_API_KEY": "cas_..." }
    }
  }
}
```

Self-hosting the analyzer? Override the endpoint with `KRAUNCHER_ANALYZER_URL`.

## Scope

v1 is deliberately one tool. The per-GPU market lineup is intentionally left
out — the agent's job is to improve its code and know the cost before running,
and a pre-run estimate (even rough or partial) is the whole point.
