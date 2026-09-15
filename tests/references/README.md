# Released forward oracles

Immutable forward bodies from the pre-unification FoldForge AF-family ports,
snapshotted 2026-09-13. File SHA256 values identify the precise local source.
Original source provenance and licenses are recorded in each model's
`SOURCE.json` and `UPSTREAM-LICENSE` (AF3 CC BY-NC-SA 4.0; Protenix/OpenDDE Apache 2.0).

Tests instantiate the current checkpoint-compatible class definitions and replace
the listed methods with these frozen bodies in an isolated test module. This
keeps the original operation ordering independent of the new composition code,
without maintaining another complete model tree or requiring an external checkout.
Unchanged primitive operators and constructors are shared; these are composition
and checkpoint-mapping tests, not an independent reimplementation of all primitives.

`attention_oracles.json` freezes the pre-MiniWorld-format Protenix/OpenDDE
`_attention` functions. Their licenses and pinned upstream source identities are
those already recorded for the corresponding ported model trees. The tests load
these bodies into isolated namespaces; they do not read a mutable "old" checkout.
