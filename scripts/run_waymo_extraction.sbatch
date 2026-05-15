#!/usr/bin/env bash
#SBATCH --job-name=waymo_extract
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --time=03:59:00
#SBATCH --ntasks=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=artifacts/phase1_waymo/slurm_%j.log
#SBATCH --error=artifacts/phase1_waymo/slurm_%j.err

set -euo pipefail

# sbatch does not source the interactive profile, so module is not on PATH by default.
source /etc/profile.d/modules.sh
source /etc/profile.d/ummodules.sh 2>/dev/null || true

module load cuda/12.6.3
module load gcc/11.2.0
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tailrisk-mp-cu126

cd /fs/nexus-projects/pc_driving/yaghoubi/tail-risk-motion-prediction

python scripts/run_difficulty_analysis.py \
  --datasets waymo \
  --val-only \
  --val-limit 5000 \
  --skip-mtr-eval \
  --skip-validate \
  --device cpu \
  --output-root artifacts/phase1_waymo \
  --subset-root data/analysis_subsets_waymo \
  --subset-cache-root data/analysis_cache_waymo
