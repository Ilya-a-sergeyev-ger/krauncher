# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""krauncher — CaS client library for submitting GPU compute tasks.

Usage::

    from krauncher import KrauncherClient

    client = KrauncherClient(api_key="cas_...", broker_url="http://...")

    @client.task(vram_gb=24, timeout=3600)
    def train(data):
        import torch
        return {"loss": 0.01}

    async def main():
        handle = await train(data={"epochs": 5})
        result = await handle
        print(result.output)
"""

import logging as _logging
import os as _os
import sys as _sys

# Load .env from CWD before anything else (does NOT override existing vars)
from ._env import load_dotenv as _load_dotenv
_load_dotenv()

_log = _logging.getLogger("krauncher")
# Own handlers below — don't propagate to root, or environments with a
# configured root handler (Colab) print every line twice.
_log.propagate = False
if not _log.handlers:
    # Progress lines (INFO) are ordinary output — stdout; Jupyter renders
    # stderr on a red background. Diagnostics stay on stderr: DEBUG (relay/CU
    # traces, visible only with KRAUNCHER_DEBUG=1) and WARNING+.
    _fmt = _logging.Formatter("%(message)s")
    _out = _logging.StreamHandler(_sys.stdout)
    _out.setFormatter(_fmt)
    _out.addFilter(lambda r: _logging.INFO <= r.levelno < _logging.WARNING)
    _log.addHandler(_out)
    _err = _logging.StreamHandler()
    _err.setFormatter(_fmt)
    _err.addFilter(lambda r: r.levelno < _logging.INFO or r.levelno >= _logging.WARNING)
    _log.addHandler(_err)
if _os.getenv("KRAUNCHER_DEBUG", "").lower() in ("1", "true", "yes"):
    _log.setLevel(_logging.DEBUG)
else:
    _log.setLevel(_logging.INFO)

from .exceptions import (
    AuthError,
    InsufficientBalanceError,
    KrauncherError,
    NoCapacityError,
    PayloadDeliveryError,
    RemoteTimeout,
    SerializationError,
    TaskError,
    TaskTimeout,
    ValueTransferError,
)
from .analyzer import TaskClassification
from .data_source import DataSource
from .KrauncherClient import KrauncherClient
from .models import Runner, TaskGroup, TaskHandle, TaskResult
from .volume import Volume

__all__ = [
    "KrauncherClient",
    "DataSource",
    "Volume",
    "TaskClassification",
    "Runner",
    "TaskGroup",
    "TaskHandle",
    "TaskResult",
    "KrauncherError",
    "AuthError",
    "InsufficientBalanceError",
    "TaskError",
    "TaskTimeout",
    "NoCapacityError",
    "PayloadDeliveryError",
    "RemoteTimeout",
    "SerializationError",
    "ValueTransferError",
]

__version__ = "0.1.2"
