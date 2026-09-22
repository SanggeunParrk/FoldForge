#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH -J ff-dump
#SBATCH --mem=96g
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -o validation/reports/logs/%x_%j.log
#SBATCH -e validation/reports/logs/%x_%j.err
set -uo pipefail
cd /home/psk6950/practice/FoldForge
source scripts/activate_env.sh
export HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=8
"$VIRTUAL_ENV/bin/python" runs/oracle-20260922/tools/dump_modules.py runs/oracle-20260922/protenix2 protenix2 --spec runs/protenix-port-20260922/5i28x1.yaml --config runs/protenix-port-20260922/protenix2/config.yaml --out runs/oracle-20260922/fold && echo OKDUMP || echo FAILDUMP
