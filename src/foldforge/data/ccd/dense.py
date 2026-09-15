"""Compatibility import for the common MiniWorld CCD preparation command.

AF3 no longer owns a separate pickle database or its own preparation pipeline.
"""

from foldforge.data.ccd.build import prepare
from foldforge.data.ccd.prepare import main

__all__ = ["main", "prepare"]

if __name__ == "__main__":
    raise SystemExit(main())
