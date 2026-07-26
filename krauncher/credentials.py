# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Storage credentials read from the caller's own environment.

Krauncher does not store credentials to third-party resources. The keys a task
needs to reach the user's S3 bucket or a private HuggingFace repo are read here
and travel to the worker inside the E2E payload, so the broker never sees them.

One set per type — one S3, one HuggingFace — not bound to any data source. The
variable names are the standard ones the AWS and HuggingFace tooling already
uses, so an environment that works with ``boto3`` works here unchanged.
"""

from __future__ import annotations

import os


def collect_credentials() -> dict[str, dict[str, str]]:
    """Read configured storage credentials, keyed by type.

    Returns only what is actually configured. An S3 key without its secret is
    skipped rather than sent half-filled — the worker would fail on it anyway,
    and a partial credential is harder to diagnose than a missing one.
    """
    creds: dict[str, dict[str, str]] = {}

    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if access_key and secret_key:
        creds["s3"] = {
            "type": "s3",
            "access_key": access_key,
            "secret_key": secret_key,
            "region": (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION", "")
            ),
            "endpoint": (
                os.environ.get("AWS_ENDPOINT_URL_S3")
                or os.environ.get("AWS_ENDPOINT_URL", "")
            ),
        }

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    )
    if token:
        creds["hf"] = {"type": "hf", "token": token}

    return creds
