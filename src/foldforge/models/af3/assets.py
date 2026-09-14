"""Compatibility entry point for the common MiniWorld CCD LMDB builder.

Prefer `foldforge ccd prepare`, which prepares every model view together.
"""

from foldforge.data.ccd.af3 import main, prepare

__all__ = ["main", "prepare"]

if __name__ == "__main__":
    raise SystemExit(main())
