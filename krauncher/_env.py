# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Minimal .env loader for krauncher.

Reads key=value pairs from a .env file into os.environ.
Does NOT override existing environment variables.

With no explicit path, the file is resolved from the CAS_CLIENT_CONFIG
environment variable (a config filename, e.g. set once in a notebook so
secrets stay out of the cells) if present, otherwise ``.env`` in the CWD.
CAS_CLIENT_CONFIG must be a real env var — it cannot live inside the file
it points to.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> int:
    """Load .env file into os.environ. Returns number of vars set."""
    if path is None:
        config = os.environ.get("CAS_CLIENT_CONFIG")
        path = Path(config) if config else Path.cwd() / ".env"
    else:
        path = Path(path)

    if not path.is_file():
        return 0

    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        # Don't override existing env vars
        if key not in os.environ:
            os.environ[key] = value
            count += 1

    return count
