#!/bin/bash
#
# Launch the NVT production chain.
# Supports both SLURM cluster and local GPU workstation execution.
#
# Usage (cluster / SLURM):
#   ./script.sh <chunks_per_job> <total_chunks> <walltime>
#
# Usage (local GPU workstation):
#   ./script.sh --local <chunks_per_job> <total_chunks>
#
# Each chunk = one execution of prod1.in (currently 200 ns).
# total_chunks controls the total sampling length (e.g. 5 chunks -> 1 us).
#
# Cluster examples:
#   * Multi-day queue (fits all 1 us in one job):
#       ./script.sh 5 5 5-00:00:00
#   * Single-day queue (chains 5 jobs back-to-back):
#       ./script.sh 1 5 1-00:00:00
#
# Local example:
#   * Run all 5 chunks sequentially on a local GPU workstation:
#       ./script.sh --local 5 5
#
# Defaults (no arguments) use the conservative single-day cluster case.

# ============================================================
# Local mode  (--local flag)
# ============================================================
if [ "${1:-}" = "--local" ]; then
    CHUNKS_PER_JOB=${2:-1}
    TOTAL_CHUNKS=${3:-5}

    # Materialise the first local worker script from run_local_template.
    sed -e "s|__START__|1|g" \
        -e "s|__CPJ__|${CHUNKS_PER_JOB}|g" \
        -e "s|__TOTAL__|${TOTAL_CHUNKS}|g" \
        run_local_template > run_NVT_1_local.sh
    chmod +x run_NVT_1_local.sh
    bash run_NVT_1_local.sh
    exit 0
fi

# ============================================================
# Cluster mode  (SLURM / sbatch)
# ============================================================
CHUNKS_PER_JOB=${1:-1}
TOTAL_CHUNKS=${2:-5}
WALLTIME=${3:-1-00:00:00}

# Materialise the first job in the chain from run_template.
sed -e "s|__START__|1|g" \
    -e "s|__CPJ__|${CHUNKS_PER_JOB}|g" \
    -e "s|__TOTAL__|${TOTAL_CHUNKS}|g" \
    -e "s|__WALLTIME__|${WALLTIME}|g" \
    run_template > run_NVT_1.cmd

sbatch run_NVT_1.cmd
