#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH -J ff-protenix-v1
#SBATCH --mem=96g
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH -o validation/reports/logs/%x_%j.log
#SBATCH -e validation/reports/logs/%x_%j.err
set -uo pipefail
cd /home/psk6950/practice/FoldForge
source scripts/activate_env.sh
ENV=$VIRTUAL_ENV
export HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8
go () {
  echo "===== $1 ====="
  "$ENV/bin/python" -c \
    "import sys; from foldforge.models.inference import run; raise SystemExit(run(sys.argv[1], sys.argv[2:]))" \
    "$2" --spec runs/protenix-port-20260922/5i28x1.yaml \
    --config "runs/protenix-port-20260922/$1/config.yaml" \
    --out "runs/protenix-port-20260922/$1" && echo "OK $1" || echo "FAILED $1"
}
go protenix-v1 protenix
go protenix1 protenix1
echo ALLDONE
