# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Value transfer — carry caller-namespace values into task kwargs and back.

The application-agnostic base for adapters (jupyter magic, editor plugins,
partner environments): named input values ride as plain JSON task kwargs and
computed values come back as the ``{name: value}`` outputs dict. The wire
format is exactly what the worker already runs and the analyzer already
classifies — numeric scalars pass through unchanged so the cas-analyzer's CU
estimator still sees ``epochs``/``batch_size``.

Anything that is not JSON-serializable (a model, a DataFrame, an open handle)
is rejected here with a clear message: move it to a data source / volume.

Values travel the E2E relay path together with the task code, so the guard
covers both each value and the total against the inline budget
(``INLINE_BUDGET_BYTES``); larger data must go through a volume / data source.
"""

from __future__ import annotations

import json
from typing import Any

from .exceptions import ValueTransferError

# Inline plaintext budget for the whole payload (code + input values), both
# directions. Set by design decision — see doc/values_api_plan.md — and to be
# adjusted after verification against the live relay.
INLINE_BUDGET_MB = 16
INLINE_BUDGET_BYTES = INLINE_BUDGET_MB * 1024 * 1024


def encode_inputs(
    names: list[str],
    namespace: dict[str, Any],
    *,
    limit_bytes: int = INLINE_BUDGET_BYTES,
) -> dict[str, Any]:
    """Fetch named input values from *namespace* as task kwargs.

    Each value must be JSON-serializable; each value and their total must fit
    *limit_bytes* (callers that also send code in the same payload pass the
    remaining budget).

    Raises:
        ValueTransferError: on a missing name, a non-JSON-safe value, or a
            budget overflow.
    """
    kwargs: dict[str, Any] = {}
    total_bytes = 0
    for name in names:
        if name not in namespace:
            raise ValueTransferError(f"input {name!r}: not defined in the caller namespace")
        value = namespace[name]
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueTransferError(
                f"input {name!r}: {type(value).__name__} is not JSON-serializable "
                f"({exc}). Pass JSON-safe values (numbers, strings, lists, dicts) "
                f"or move large/complex data to a data source / volume."
            ) from exc
        size = len(encoded.encode("utf-8"))
        if size > limit_bytes:
            raise ValueTransferError(
                f"input {name!r}: {size / (1024 * 1024):.1f} MB exceeds the "
                f"{limit_bytes / (1024 * 1024):.1f} MB inline budget — "
                f"use a volume / data source."
            )
        total_bytes += size
        kwargs[name] = value

    if total_bytes > limit_bytes:
        raise ValueTransferError(
            f"inputs total {total_bytes / (1024 * 1024):.1f} MB exceeds the "
            f"{limit_bytes / (1024 * 1024):.1f} MB inline budget — "
            f"move large values to a volume / data source."
        )
    return kwargs


def decode_outputs(output: Any, names: list[str]) -> dict[str, Any]:
    """Pull the task's returned outputs dict back into named values.

    Raises:
        ValueTransferError: when the task returned something other than the
            outputs dict, or a requested name is missing from it.
    """
    if not isinstance(output, dict):
        raise ValueTransferError(
            f"task returned {type(output).__name__}, expected the outputs "
            f"dict — was the submitted code altered?"
        )
    values: dict[str, Any] = {}
    for name in names:
        if name not in output:
            raise ValueTransferError(f"output {name!r}: missing from the task result")
        values[name] = output[name]
    return values
