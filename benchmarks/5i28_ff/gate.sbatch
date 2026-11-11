#!/bin/bash
# A CPU-only gate: returns when none of YOUR jobs whose name matches WAIT_FOR (a grep -E pattern, default '^b5i28' = the 5I28 kit jobs) is left in the
# queue, polling every 60 s. Jobs submitted with --dependency=afterany:<this job> then start only after the kit measurements are over (a timing must
# not share a node's GPUs / CPUs with another measurement).
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=normal_h100
#SBATCH --gres=none
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH -o slurm-%x-%j.log
WAIT_FOR=${WAIT_FOR:-^b5i28}
echo "GATE start $(date -Is) waiting for jobs matching '$WAIT_FOR'"
while :; do
  left=$(squeue -u "$USER" -h -o "%j" | grep -cE "$WAIT_FOR")
  [ "$left" = 0 ] && break
  sleep 60
done
echo "GATE open $(date -Is)"
