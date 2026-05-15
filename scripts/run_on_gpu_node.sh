#!/usr/bin/env bash
# Reproducible GPU-node launcher for tail-risk-motion-prediction.
#
# This script is intended to be invoked from the LOGIN node. It:
#   1. allocates a GPU-enabled SLURM session via srun (interactive or batch-style),
#   2. loads the required CUDA/GCC modules,
#   3. activates the conda environment,
#   4. cd's into the project root,
#   5. runs either an interactive shell or the command passed as arguments.
#
# Usage examples:
#   # interactive shell with the environment ready
#   bash scripts/run_on_gpu_node.sh
#
#   # one-shot: run the Phase 1 feature extraction
#   bash scripts/run_on_gpu_node.sh \
#     python scripts/run_difficulty_analysis.py \
#       --datasets av2 --mtr-train --val-only --device cuda
#
#   # adjust SLURM knobs from the caller:
#   SLURM_TIME=03:59:00 SLURM_MEM=32gb bash scripts/run_on_gpu_node.sh ...
#
# SLURM parameters are taken from the Nexus/gamma defaults documented in README.md.

set -euo pipefail

SLURM_QOS="${SLURM_QOS:-huge-long}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-gamma}"
SLURM_PARTITION="${SLURM_PARTITION:-gamma}"
SLURM_TIME="${SLURM_TIME:-01:59:00}"
SLURM_NTASKS="${SLURM_NTASKS:-4}"
SLURM_MEM="${SLURM_MEM:-16gb}"
SLURM_GRES="${SLURM_GRES:-gpu:rtxa5000:1}"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.3}"
GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
CONDA_ENV="${CONDA_ENV:-tailrisk-mp-cu126}"

REPO_ROOT="${REPO_ROOT:-/fs/nexus-projects/pc_driving/yaghoubi/tail-risk-motion-prediction}"

# Everything after `--` (or all positional args) is what we run on the node.
if [[ "$#" -eq 0 ]]; then
  REMOTE_CMD="zsh"
else
  # quote-preserving join
  REMOTE_CMD="$(printf ' %q' "$@")"
  REMOTE_CMD="${REMOTE_CMD:1}"
fi

INNER_SCRIPT=$(cat <<EOF
set -euo pipefail
module load ${CUDA_MODULE}
module load ${GCC_MODULE}
# shellcheck disable=SC1091
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
cd ${REPO_ROOT}
${REMOTE_CMD}
EOF
)

exec srun --pty \
  --qos="${SLURM_QOS}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --time="${SLURM_TIME}" \
  --ntasks="${SLURM_NTASKS}" \
  --mem="${SLURM_MEM}" \
  --gres="${SLURM_GRES}" \
  bash -c "${INNER_SCRIPT}"
