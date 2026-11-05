# Python API

Load a model and its inputs independently. Inference requires CUDA, an available
checkpoint and the [CCD database](MINIWORLD-FORMAT.md). Loading performs no weight
download, MSA search or template search.

```python
import foldforge as ff

inputs = ff.load_inputs("input.yaml")
with ff.load_model("boltz2", config="config.yaml", device="cuda:0") as boltz:
    result = boltz(inputs)

coords = result.coords       # CPU [samples, atoms, 3], angstroms
plddt = result.plddt         # CPU token confidence [0, 1], or None
cif = result.cifs[0]         # mmCIF text
saved = result.save("./results/boltz2")
```

The predictor loads its structure weights and execution wrappers once. The same
predictor accepts different targets; the same `Inputs` may be passed to different
predictors. Conditioning must be supported by each selected checkpoint. Features
are prepared separately per model without modifying the caller's inputs.

## Trunk reuse and test-time scaling

`run_trunk()` prepares the input, runs the requested total number of trunk passes,
and returns a `TrunkState` holding residue representations and prepared diffusion
inputs on the model's GPU. `sample_diffusion()` runs diffusion and confidence/output
heads from that state; it performs no input featurization, language-model forward
or trunk rerun.

```python
import foldforge as ff

inputs = ff.load_inputs("input.yaml")
with ff.load_model("boltz2", config="config.yaml", device="cuda:0") as boltz:
    with boltz.run_trunk(inputs, recycles=10, seed=0) as trunk:
        for seed in range(10):
            result = boltz.sample_diffusion(
                trunk,
                samples=5,
                steps=200,
                seed=seed,
            )
            result.save(f"./results/seed-{seed}")
```

This produces 50 diffusion samples from one trunk. Small sampling batches keep GPU
memory bounded while the returned CPU results can be saved, scored or retained.
The loop uses independent diffusion seeds. Repeating the same seed requests the
same random stream. `recycles` always means total trunk passes, including the first.

For multiple trunk candidates, call `run_trunk()` again with another seed or pass
count, then sample each state. Retaining multiple states consumes GPU memory; close
states when their candidates are no longer needed. Neither a later trunk run nor a
sampling override changes the predictor's loaded `config`.

```python
import foldforge as ff

inputs = ff.load_inputs("input.yaml")
with ff.load_model("opendde", config="config.yaml") as model:
    for trunk_seed in range(3):
        with model.run_trunk(inputs, seed=trunk_seed, recycles=10) as trunk:
            result = model.sample_diffusion(trunk, seed=0, samples=5)
            result.save(f"./results/trunk-{trunk_seed}")
```

Each state belongs to the exact predictor that created it, including its checkpoint,
mode, precision and CUDA device. Passing it to another predictor raises
`ff.TrunkStateError`. A closed state cannot be sampled. Closing the predictor also
closes its surviving states. CPU results remain valid after either is closed.
`trunk.embeddings` exposes a read-only mapping of residue tensors for inspection;
the tensors themselves are borrowed and must not be mutated. `trunk.report` records
the trunk seed, pass count and execution observations. Sampling reports include
`trunk_reused`, `trunk_execution`, actual sample/step counts and the diffusion seed.
Timings exclude idle time between trunk preparation and a later sampling request.

The basic `model(inputs)` call and the staged API use the same network stage
implementations, including OpenDDE structural-token expansion/restoration. Staged
inference supports `execution.scope: denoiser` and `trunk`. The experimental whole
model compile scope is available through `model(inputs)` and cannot be split.
The [measured compiled-repeat limit](EXECUTION-SCOPES.md#repeated-compiled-predictions)
remains applicable when compilation is explicitly enabled; eager and graph-only
long repeats passed the recorded reproducibility checks.

## Inputs and configuration

`load_inputs(source, *, base_dir=None)` accepts a MiniWorld YAML path, AF3 JSON
path, AF3 mapping, an existing `Inputs`, or a lower-level resolved `Input`. JSON
chains and referenced MSA/template resources are validated using the AF3 parser.
Relative paths refer to the input file's directory, or `base_dir` for a mapping.
Resources read at input load time are retained for reuse. Missing protein
conditioning fields mean empty conditioning, with no search.

For a protein without conditioning:

```python
import foldforge as ff

inputs = ff.load_inputs(ff.protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", name="example"))
config = ff.Config.model_validate({
    "precision": "bf16",
    "trunk": {"recycles": 10},
    "diffusion": {"steps": 200},
})
with ff.load_model("boltz2", config=config) as model:
    result = model(inputs, samples=5)
```

`config` accepts `ff.Config`, a mapping, a YAML path or `None` for defaults. It
contains execution policy; the model name stays an explicit `load_model()` argument.
Omitted call options preserve their defaults. YAML sample counts come from the
input spec unless `samples=` overrides them; JSON/mapping inputs default to one.
All counts must be positive integers. Unknown config fields are rejected.

| Operation | Options |
| --- | --- |
| `ff.load_model(name, config=None)` | `checkpoint`, `device`, `engine_cache_dir` |
| `model(inputs)` | `samples`, `out`, `ccd_db`, `base_dir`, `verbose` |
| `model.run_trunk(inputs)` | `recycles`, `seed`, `ccd_db`, `base_dir`, `verbose` |
| `model.sample_diffusion(trunk)` | `samples`, `steps`, `seed`, `out` |

`model.config` returns a detached copy of the loaded policy. The structure network
is eager-loaded; auxiliary ESM2/ESM-C towers and per-release shims load on first use
and stay with the predictor until `close()`. There is no global model pool.

## Device, lifetime and concurrency

`device="cuda:1"` pins a predictor to that device for loading, featurization,
inference and sampling. Omitting the index selects the current CUDA device once
at load time. Later changes to the caller's device context cannot redirect it;
that context is restored after every operation. CPU inference is unsupported.

Use context managers or explicit `model.close()` / `trunk.close()`. Closing releases
owned references; normal Torch/engine caches can still retain reserved GPU memory.
The objects are process-bound and are not portable serialized inference jobs.

Overlapping operations on the same predictor or state raise
`ff.ConcurrentCallError` before preparation. Different predictors may be called
from threads, but their operations serialize because engine policy, chemistry
scopes, matmul precision and RNGs are process-global. Use worker processes for
parallel predictions across GPUs. Caller settings and RNGs restore on failures too.
Library calls raise Python exceptions instead of exiting the process. Lifecycle and
input errors have public `ff.FoldForgeError` subclasses; missing resources retain
standard filesystem exceptions and invalid configurations use Pydantic validation.

## Saving and the CLI

Model calls and staged calls return CPU results without prediction artifacts by
default. Compilation and autotuning may still write their normal caches.
`result.save(path)` and the library's `out=path` accept a caller-owned directory,
including an absolute path outside FoldForge's cache. Relative paths are relative
to the working directory. `save()` returns a new result with an output manifest;
it leaves the original result unchanged. `out=` additionally saves resolved input
and configuration artifacts during the inference request.

`ff.predict(inputs, config, model="boltz2", device="cuda:0")` is the one-shot
convenience function: it opens a predictor, runs the same pipeline and closes it.
For raw tensor batches, the lower-level `ff.load()` network loader remains available.

The CLI chooses a run inside `$FOLDFORGE_RUNS` (default `$FOLDFORGE_HOME/runs`) before
calling the library. Its path containment rule is separate from library saving.
Explicit CLI flags override config values; omitted flags preserve YAML/defaults.

```bash
foldforge models
foldforge fold --help
foldforge fold boltz2 --spec input.yaml --config config.yaml --device cuda:1 --out my-run
foldforge fold boltz2 --spec input.yaml --config config.yaml --no-compile --recycles 10
```

`models --json` and `models --plain` are available for scripts. Legacy
`--input input.json --checkpoint weights.pt` is retained, with five samples by
default. Both input forms use the public inference lifecycle.
