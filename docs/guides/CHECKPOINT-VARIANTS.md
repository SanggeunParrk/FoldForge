# Checkpoint variants

Currently, the following models are supported. Unless specified otherwise,
models are trained based on the 2021-09-30 wwPDB cutoff.

|           Model Name                |  ESM/MSA/Constraint/RNA MSA/Template | Model Parameters (M) |
|-------------------------------------|--------------------------------------|----------------------|
| `protenix_base_default_v0.5.0`      |      ❌ / ✅ / ❌ / ❌ / ❌            |         368.09       |
| `protenix_base_constraint_v0.5.0`   |      ❌ / ✅ / ✅ / ❌ / ❌            |         368.30       |
| `protenix_mini_esm_v0.5.0`          |      ✅ / ✅ / ❌ / ❌ / ❌            |         135.22       |
| `protenix_mini_ism_v0.5.0`          |      ✅ / ✅ / ❌ / ❌ / ❌            |         135.22       |
| `protenix_mini_default_v0.5.0`      |      ❌ / ✅ / ❌ / ❌ / ❌            |         134.06       |
| `protenix_tiny_default_v0.5.0`      |      ❌ / ✅ / ❌ / ❌ / ❌            |         109.50       |

The following models support inference with templates and RNA MSA.
Format: protenix_{model_size}_{features}_{version}
| `protenix_base_default_v1.0.0`      |      ❌ / ✅ / ❌ / ✅ / ✅            |         368.48       |

For practical application scenarios, `protenix_base_20250630_v1.0.0` is trained based
on the 2025-06-30 wwPDB cutoff
and is also released to the community.
For fair benchmarks of model improvements across different versions, please use
`protenix_base_default_v1.0.0`.

| `protenix_base_20250630_v1.0.0`     |      ❌ / ✅ / ❌ / ✅ / ✅            |         368.48       |

Scaled-up models
| `protenix-v2`       |      ❌ / ✅ / ❌ / ✅ / ✅            |         464.44       |
