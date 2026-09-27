"""Run the released IntelliFold runner, with numpy scalars JSON-serialisable.

Its summary writer passes numpy float32 to json.dump, which this environment's
numpy refuses; the model and every tensor are untouched.
"""

import json
import runpy
import sys
from typing import Any

import numpy as np

_default = json.JSONEncoder.default


def default(self: json.JSONEncoder, o: Any) -> Any:
    """Serialise numpy values the release hands to json.dump."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return _default(self, o)


json.JSONEncoder.default = default
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
