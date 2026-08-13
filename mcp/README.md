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

## The `estimate_gpu_time_and_cost` tool

Input: `code` — the job's Python source. Send the **whole module**, not just
the function that trains: the model, the dataset size and the step count are
read off whatever you pass, so a helper that builds the model or loads the data
takes its share of the answer with it when it is left behind. Extra code costs
nothing — anything that is not the GPU job is simply not detected. (A
`@client.task`-decorated function is fine, and so is a plain script.)

Optional: `run_args` — the arguments the job will be called with, when the
source leaves them open (`{"epochs": 3, "batch_size": 32}` for a
`def train(epochs, batch_size)`). Names must match parameters some function in
the source declares; scalars only. Pass what the run actually is, never a
guess — the estimate scales with these. `findings` reports which arguments were
applied, or that none matched.

Output, on the reference card (**RTX PRO 6000 WS**, the card CU is normalized
to):

```jsonc
{
  "reference_card": "RTX PRO 6000 WS",
  "compute_sec": 20.2,      // the three phases of wall time on the ref card
  "setup_sec": 3.0,
  "io_sec": 2.1,
  "min_vram_gb": 6,         // estimated requirement + a 5% safety margin
  "min_disk_gb": 10,
  "confidence": 1.0,        // 0-1
  "analysis_method": "ast", // how the estimate was reached; "ast" is the plain case
  "cpu_only": false,
  "spread": 1.51,           // slow end of the measured host population
  "spread_reason": "1.51x { cv_training, nw=one } on the compute phase, observed on 42 runs across 16 hosts (worst 2.45x)",
  "calibration_basis": "calibrated",  // | "extrapolated" | "uncalibrated"
  "iterations": 4690,       // the step count the estimate scales with
  "iteration_basis": "literal_loop",  // read from the code, or assumed
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

`analysis_method` names the route the estimate took. `ast` is the plain case:
the code was read and that was enough. Anything longer is the analyzer saying
it wanted a second opinion and reports what happened instead —
`ast_only_llm_disabled`, `ast_only_llm_unavailable`, `ast_degraded_queue_full`,
`ast_only_llm_failed`, `ast+tree`. Those come with a lowered `confidence`;
treat the value as prose, not as an enum to switch on.

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

`spread` and `calibration_basis` answer a different question than `confidence`.
Confidence is about the reading of the code; spread is about the world the code
will run in — the same source on the same card lands over a range of hosts, and
1.51 means the slow end of that measured population takes about half again as
long as the estimate. Both can be high at once, which is the honest description
of a job whose time the GPU does not govern: the card waits on the host, so the
run inherits whichever host it lands on. The lever there is GPU utilization —
whatever leaves the card idle in this code (data loaded in the main process,
per-item preprocessing, synchronous transfers, a batch too small to fill the
card). `spread_reason` names the population the number came from and how many
runs it rests on; `calibration_basis` says whether this shape was measured at all
(`calibrated`), answered by a neighbour (`extrapolated`), or matched nothing
(`uncalibrated` — read the seconds as an order of magnitude).

`iterations` is the step count the whole estimate scales with, and
`iteration_basis` says where that count came from — the one thing a wrong
estimate is most often wrong about. Read from the code: `max_steps`,
`literal_loop`, `epochs_x_samples`, `llm_decode`, `diffusion_steps`. Assumed,
because the code did not say: `epochs_x_default_samples` (a typical dataset
size for that model stood in), `dataset_size_estimate` (steps derived from the
dataset's byte size), `unknown` (a single step assumed). On an assumed basis
the seconds move with the assumption and can be wrong by orders of magnitude —
state the real number in the source and estimate again.

What it does **not** return: the cost model's calibration coefficients or
weights. Only what the analyzer detected in the code leaves the server.

## Install

```sh
pip install -e .        # from this directory; also installs the analyzer client
```

An API key is **optional**. Without one the server calls the public analyzer
**keyless**, under a per-IP daily quota (when the quota is reached, the tool
returns a short note to register for a larger one). Set a key to use your own
account and skip the quota:

```sh
export KRAUNCHER_API_KEY=cas_...   # optional
```

The key decides **which analyzer answers, never what is asked**: the request is
built the same way with or without it, so the same code cannot come back with
two different estimates. A key that cannot resolve an analyzer — revoked, or the
broker is unreachable — falls back to the keyless route rather than failing the
estimate.

Verify it works without wiring up a client — runs the tool on a sample task
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
    "gpu-estimator": {
      "command": "krauncher-mcp"
    }
  }
}
```

To use your own account (keyed, exempt from the per-IP quota), add the key:

```jsonc
{
  "mcpServers": {
    "gpu-estimator": {
      "command": "krauncher-mcp",
      "env": { "KRAUNCHER_API_KEY": "cas_..." }
    }
  }
}
```

Self-hosting the analyzer? Override the endpoint with `KRAUNCHER_ANALYZER_URL`.

Nothing else to configure for discovery: the tool ships marked
`anthropic/alwaysLoad`, so on hosts that defer MCP schemas behind a tool search
(Claude Code does this by default) it is in context from the first turn rather
than waiting to be searched for. One tool, one small schema — that is the whole
budget it spends.

## Scope

v1 is deliberately one tool. The per-GPU market lineup is intentionally left
out — the agent's job is to improve its code and know the cost before running,
and a pre-run estimate (even rough or partial) is the whole point.
