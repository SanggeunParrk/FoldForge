"""Compatibility entry point; implementation lives in the installed package."""

from foldforge.models.esmfold2.inference import main

if __name__ == "__main__":
    raise SystemExit(main())
