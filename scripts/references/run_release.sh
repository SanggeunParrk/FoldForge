#!/bin/bash
# Fold one reference input with a family's RELEASED implementation.
#
#   run_release.sh FAMILY TARGET SEED OUT_DIR
#
# Conditions shared by every family, so the references compare like with like:
# no template, no MSA unless the target ships one (<target>/A.a3m, then every
# family reads that same alignment), four trunk passes (Chai-1: its default three), 200
# diffusion steps, five samples, one model seed. Protenix and IntelliFold run
# at fp32, as their defaults do not.
#
# The released environments and their dependency trees are too large for the
# repository. FOLDFORGE_RELEASE_ROOT points at the directory that holds them
# (see references/README.md for how each was built); the Python environments of
# Boltz, Protenix and Chai-lab live under /public_data/thalkak_envs.
set -euo pipefail
FAMILY=$1; TARGET=$2; SEED=$3; OUT=$(realpath -m "$4")
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REL=${FOLDFORGE_RELEASE_ROOT:-$ROOT/releases}
ENVS=${FOLDFORGE_RELEASE_ENVS:-/public_data/thalkak_envs}
IN=$ROOT/references/inputs/$TARGET
HERE=$ROOT/scripts/references
mkdir -p "$OUT"
# Inputs may name their MSA as {IN}/A.a3m; materialise absolute paths per run.
MAT=$OUT/_input; mkdir -p "$MAT"
for f in "$IN"/*; do sed "s#{IN}#$MAT#g" "$f" > "$MAT/$(basename "$f")"; done
IN=$MAT
USE_MSA=false; ls "$IN"/*.a3m >/dev/null 2>&1 && USE_MSA=true
cd "$ROOT"
source scripts/activate_env.sh

case $FAMILY in
boltz2)
  cd "$REL" && "$ENVS/boltz/bin/boltz" predict "$IN/boltz.yaml" --out_dir "$OUT" \
    --cache "$REL/boltz-cache" --model boltz2 --recycling_steps 3 --sampling_steps 200 \
    --diffusion_samples 5 --seed "$SEED" --output_format mmcif --override ;;
protenix1|protenix2)
  E=protenix; M=protenix_base_default_v1.0.0
  [ "$FAMILY" = protenix2 ] && E=protenix_v2 && M=protenix-v2
  cd "$REL" && PROTENIX_ROOT_DIR=$REL/protenix-root LAYERNORM_TYPE=torch "$ENVS/$E/bin/protenix" pred \
    -i "$IN/protenix.json" -o "$OUT" -s "$SEED" -c 4 -p 200 -e 5 -n "$M" --use_msa "$USE_MSA" \
    -d fp32 --trimul_kernel torch --triatt_kernel torch ;;
openfold3|openfold3-preview2)
  S=$ROOT/../refs/uplifting-biomolecular-modeling/openfold3_ob0/stock/src
  K=$ROOT/model_checkpoints/openfold3/release/of3-ob-2025-06-30-174k.pt
  if [ "$FAMILY" = openfold3-preview2 ]; then
    S=${OF3_PREVIEW2_SRC:-/home/hwlee/project/openfold3}; K=${OF3_PREVIEW2_CKPT:-/home/hwlee/.openfold3/of3-p2-155k.pt}
  fi
  # OpenFold3 parses only MSA files whose names it knows; colabfold_main is one.
  [ -f "$IN/A.a3m" ] && ln -sf "$IN/A.a3m" "$IN/colabfold_main.a3m"
  python - "$IN/of3.json" "$SEED" "$OUT/query.json" <<'PY'
import json, sys
query = json.load(open(sys.argv[1]))
query["seeds"] = [int(sys.argv[2])]
json.dump(query, open(sys.argv[3], "w"), indent=1)
PY
  PYTHONPATH=$REL/of3-deps:$S python "$S/openfold3/run_openfold.py" predict --query-json "$OUT/query.json" \
    --use-msa-server=False --inference-ckpt-path "$K" --num-diffusion-samples 5 --output-dir "$OUT" ;;
rosettafold3)
  D=$REL/rf3-deps; S=$D/_src/foundry-4010e3e2e
  PYTHONPATH=$D:$S/src:$S/models/rf3/src python -c "from rf3.cli import app; app()" fold \
    ckpt_path="$ROOT/model_checkpoints/rosettafold3/release/rf3_foundry_01_24_latest_remapped.ckpt" \
    inputs="$IN/rf3.json" out_dir="$OUT" n_recycles=4 num_steps=200 diffusion_batch_size=5 \
    early_stopping_plddt_threshold=0 seed="$SEED" ;;
intellifold2)
  PYTHONPATH=$REL/if2-deps python "$HERE/intellifold_run.py" "$REL/if2-deps/runner/intellifold_inference.py" \
    "$IN/intellifold.yaml" --out_dir "$OUT" --cache "$ROOT/model_checkpoints/intellifold2/release" \
    --model v2 --seed "$SEED" --recycling_iters 3 --num_diffusion_samples 5 --sampling_steps 200 \
    --precision no --num_workers 0 --override ;;
chai1)
  CHAI_DOWNLOADS_DIR=$ROOT/model_checkpoints/chai1 \
  PYTHONPATH=$REL/chai-rdkit-deps:$REL/chai-msa-deps:$REL/chai-deps:$ENVS/chai-lab \
    python "$HERE/chai_run.py" "$IN/chai.fasta" "$OUT" "$SEED" ;;
opendde)
  S=${OPENDDE_SRC:-/home/kkh517/OpenDDE}
  cd "$S" && OPENDDE_ROOT_DIR=$REL/protenix-root LAYERNORM_TYPE=torch PYTHONPATH=$S \
    "$ENVS/protenix_v2/bin/python" runner/batch_inference.py pred -i "$IN/protenix.json" -o "$OUT" \
    -s "$SEED" -c 4 -p 200 -e 5 -n opendde_v1 --use_msa "$USE_MSA" -d fp32 \
    --trimul_kernel torch --triatt_kernel torch \
    --load_checkpoint_path "$ROOT/model_checkpoints/opendde/opendde.pt" ;;
esmfold2)
  # The Biohub Transformers fork that carried ESMFold2/ESM-C was taken down;
  # the references use a copy of it kept beside the other release trees.
  PYTHONPATH=$REL/esmfold2-transformers-fork \
    python "$HERE/esmfold2_run.py" "$IN/boltz.yaml" "$OUT" "$SEED" ;;
*) echo "unknown family $FAMILY" >&2; exit 2 ;;
esac
