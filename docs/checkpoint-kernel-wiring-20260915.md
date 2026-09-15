# Checkpoint kernel wiring (2026-09-15)

The authoritative shape census and build table are in miniworld-engine:
`docs/checkpoint-shapes-20260915.md`, its JSON companion, and
`src/miniworld_engine/kernels/registry_module.csv`.

This change adds inference wiring for sigmoid gate + output projection in
AF-family/shared MSA attention, AF3 atom LN + pair projection, eligible BF16
atom coordinate LN + projection (128→3), AF3 unconditioned
SwiGLU, and ESMFold2 atom RMSNorm/modulation plus SwiGLU. Existing square attention,
TriMul, conditional transition, AdaLN and SWA QK/RoPE wiring is retained.

Native precision means BF16 projection parameters and FP32 norm parameters;
FP32 coordinate-derived activations keep their fallback. PyTorch and cuEquivariance
remain separately selectable. Residual additions and the outer conditioning gate
retain their existing owners. ESM atom modulation uses a raw gate, not sigmoid,
and preserves the dtype-dependent RMSNorm epsilon.

The rectangular 32×128 atom attention core and Protenix-v1's asymmetric 64/128
template TriMul still require their reference equations. Connecting surrounding
kernels does not imply that these two whole operations are engine-supported.

Tests: `FoldForge/tests/kernels/test_checkpoint_fusions.py` and
`FoldForge/tests/kernels/test_backend_attention.py`. Engine build/API regression tests and
the channel census are recorded in the engine report. These are module-level
checks; they are not a new end-to-end timing table or a completed all-GPU retune.

LayerNorm + linear callers share `layernorm_projection`. Only narrow output widths
(up to 16) fuse: A100 compile+CUDA-graph measurements found the tested wide output
projections 1.9–5.6 times slower with the fused kernel. Those keep their existing
standalone engine norm plus linear. The coordinate projection microbenchmark was
0.015913 ms fused versus 0.017797 ms existing; this is not an end-to-end speedup.

The executable engine shape registry now has 141 rows (previously 53), including
native FP32 norm affine tensors and token output norm augmentation 5/48. Added
shapes still require a GPU-specific cache build. See the engine validation report
for exact test coverage and exclusions.
