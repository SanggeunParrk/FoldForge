"""The ONLY place FoldForge touches miniworld-engine's private modules (``kernels._capture``, ``kernels._tf32``, the ``_PACKS*`` stores of
``kernels.integrations.*``). Everything else imports from here, so that when the engine exports a public API
(``static_weights`` / ``static_inputs`` / ``clear_weight_caches``, requested in docs/ENGINE-REQUESTS.md) this file is the one to change.
"""

from __future__ import annotations

import contextlib
import sys

_PACK_STORES = ("_PACKS", "_PACKS32")


def static_weights() -> contextlib.AbstractContextManager[None]:
    """The weights stay fixed between graph replays (inference): the engine may serve its eager weight re-layouts to a capture."""
    from miniworld_engine.kernels import _capture

    return _capture.static_weights()


def static_inputs() -> contextlib.AbstractContextManager[None]:
    """The pair and conditioning inputs of the captured callable keep their contents for the life of the graph."""
    from miniworld_engine.kernels import _capture

    return _capture.static_inputs()


def clear_weight_caches() -> None:
    """Drop the engine's eager weight re-layout caches before weights are replaced.

    The H100 kernels keep weight forms (``w.t().contiguous()``, TF32-rounded triples, packed blocks) keyed by tensor identity and version; a
    new weight allocated at the address of a freed one must not be served the old copy. A process that builds a second model, or a test
    that builds many, clears them first."""
    from miniworld_engine.kernels import _capture

    _capture._FORMS.clear()
    tf32 = sys.modules.get("miniworld_engine.kernels._tf32")
    if tf32 is not None:
        tf32._CACHE.clear()
    for name, module in list(sys.modules.items()):
        if name.startswith("miniworld_engine.kernels.integrations."):
            for attr in _PACK_STORES:
                store = getattr(module, attr, None)
                if isinstance(store, dict):
                    store.clear()
