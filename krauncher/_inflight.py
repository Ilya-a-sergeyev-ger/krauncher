# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""In-flight task registry — cancel submitted tasks on interpreter exit.

Every real broker submission registers its :class:`TaskHandle` here; reaching
a terminal status (or an explicit cancel) unregisters it. :func:`cancel_all`
sweeps whatever is left through ``TaskHandle._cancel_remote()`` — the same
best-effort synchronous broker DELETE + relay CancelTask used for
cancel-on-abandon in ``wait()``.

The sweep is NOT installed for plain scripts — fire-and-forget submission
keeps working. Inside an IPython kernel it is registered with ``atexit`` on
first submission, so "Restart the kernel" cancels everything the notebook
submitted and did not await to completion.

The registry is keyed by handle identity, not task_id — ``wait()`` can
transparently resubmit on no_capacity, which changes the handle's task_id.
"""

from __future__ import annotations

import atexit
import logging
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TaskHandle

logger = logging.getLogger("krauncher")

_lock = threading.Lock()
_inflight: dict[int, "TaskHandle"] = {}
_hook_installed = False


def register(handle: "TaskHandle") -> None:
    with _lock:
        _inflight[id(handle)] = handle
    _maybe_install_ipython_hook()


def unregister(handle: "TaskHandle") -> None:
    with _lock:
        _inflight.pop(id(handle), None)


def cancel_all() -> int:
    """Cancel every registered task not yet at a terminal status.

    Synchronous and best-effort by design — it runs from atexit, where no
    event loop is available. Safe to call repeatedly. Returns the number of
    tasks swept.
    """
    from .models import TERMINAL_STATUSES

    with _lock:
        handles = list(_inflight.values())
        _inflight.clear()

    swept = 0
    for handle in handles:
        if handle._result is not None or handle._last_status in TERMINAL_STATUSES:
            continue
        logger.info("cancelling in-flight task on exit: %s", handle.task_id)
        handle._cancel_remote()
        swept += 1
    return swept


def _maybe_install_ipython_hook() -> None:
    """Register the atexit sweep — but only inside an IPython kernel.

    Detection is passive: if IPython was never imported into this process,
    there is no kernel and nothing to install.
    """
    global _hook_installed
    if _hook_installed or "IPython" not in sys.modules:
        return
    try:
        if sys.modules["IPython"].get_ipython() is None:
            return
    except Exception:
        return
    atexit.register(cancel_all)
    _hook_installed = True
    logger.debug("in-flight atexit sweep installed (IPython kernel detected)")
