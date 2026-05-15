#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-tailrisk-mp-cu126}"
CONDA_ROOT="${CONDA_ROOT:-/fs/nexus-scratch/yaghoubi/anaconda3}"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/day1"
SUBSET_ROOT="${REPO_ROOT}/data/smoke_subsets"

if [[ -n "${SLURM_PROCID:-}" && "${SLURM_PROCID}" != "0" ]]; then
  exit 0
fi

mkdir -p "${ARTIFACT_DIR}"
mkdir -p "${REPO_ROOT}/tmp/mplconfig"
export MPLCONFIGDIR="${REPO_ROOT}/tmp/mplconfig"

bash "${REPO_ROOT}/scripts/setup_day1_env.sh" "${ENV_NAME}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python "${REPO_ROOT}/scripts/create_scenarionet_subset.py" \
  --source-root /fs/nexus-projects/pc_driving/datasets/sn_womd \
  --dest-root "${SUBSET_ROOT}/waymo_smoke" \
  --split training:validation=validation_0 \
  --split validation:validation=validation_0 \
  --limit 32 \
  --clear

python "${REPO_ROOT}/scripts/create_scenarionet_subset.py" \
  --source-root /fs/nexus-projects/pc_driving/datasets/argoverse2_sn \
  --dest-root "${SUBSET_ROOT}/av2_smoke" \
  --split train:val=val_0 \
  --split val:val=val_0 \
  --limit 32 \
  --clear

python "${REPO_ROOT}/scripts/day1_smoke_test.py" \
  --repo-root "${REPO_ROOT}" \
  --waymo-train "${SUBSET_ROOT}/waymo_smoke/training" \
  --waymo-val "${SUBSET_ROOT}/waymo_smoke/validation" \
  --waymo-cache-train /fs/nexus-projects/pc_driving/datasets/waymo_cache/training/sn_womd \
  --waymo-cache-val /fs/nexus-projects/pc_driving/datasets/waymo_cache/validation/sn_womd \
  --av2-train "${SUBSET_ROOT}/av2_smoke/train" \
  --av2-val "${SUBSET_ROOT}/av2_smoke/val" \
  --av2-cache-train /fs/nexus-projects/pc_driving/datasets/argoverse_cache/train/argoverse2_sn \
  --av2-cache-val /fs/nexus-projects/pc_driving/datasets/argoverse_cache/val/argoverse2_sn \
  --waymo-subset-cache "${ARTIFACT_DIR}/subset_cache_waymo" \
  --av2-subset-cache "${ARTIFACT_DIR}/subset_cache_av2" \
  --report-path "${ARTIFACT_DIR}/day1_report.json"

echo "Day-1 report written to ${ARTIFACT_DIR}/day1_report.json"
