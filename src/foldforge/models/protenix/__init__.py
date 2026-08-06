"""Protenix v1 and v2 (ByteDance).

**Not ported.** Upstream ships its own weight downloader.

One package for both versions rather than ``protenix_v1`` / ``protenix_v2``:
they share most of the stack, and splitting them would duplicate the parts that
are identical while hiding the parts that are not. Select the version at load
time and keep whatever genuinely differs in its own module, so the diff between
v1 and v2 stays readable — that difference is itself worth having, since running
both from one codebase is a reason this repo exists.
"""
