#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-tailrisk-mp-cu126}"
CONDA_ROOT="${CONDA_ROOT:-/fs/nexus-scratch/yaghoubi/anaconda3}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.3}"
GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/day1"
UNITRAJ_DIR="${REPO_ROOT}/third_party/UniTraj"
UNITRAJ_REMOTE="${UNITRAJ_REMOTE:-https://github.com/vita-epfl/UniTraj.git}"
UNITRAJ_FALLBACK_SOURCE="${UNITRAJ_FALLBACK_SOURCE:-/fs/nexus-projects/pc_driving/yaghoubi/UniTraj}"
SCENARIONET_DIR="${REPO_ROOT}/third_party/ScenarioNet"
SCENARIONET_REMOTE="${SCENARIONET_REMOTE:-https://github.com/metadriverse/scenarionet.git}"
SCENARIONET_FALLBACK_SOURCE="${SCENARIONET_FALLBACK_SOURCE:-/fs/nexus-projects/pc_driving/yaghoubi/pc_driving/third_party/scenarionet}"

mkdir -p "${ARTIFACT_DIR}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

module load "${CUDA_MODULE}"
module load "${GCC_MODULE}"

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME
  CUDA_HOME="$(cd "$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")" && pwd)"
fi

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

if [[ ! -d "${UNITRAJ_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${UNITRAJ_DIR}")"
  if ! git clone "${UNITRAJ_REMOTE}" "${UNITRAJ_DIR}"; then
    rm -rf "${UNITRAJ_DIR}"
    git clone "${UNITRAJ_FALLBACK_SOURCE}" "${UNITRAJ_DIR}"
  fi
fi

python -m pip install -r "${UNITRAJ_DIR}/requirements.txt"
python -m pip install \
  metadrive-simulator==0.4.2.3 \
  nuscenes-devkit

if [[ ! -d "${SCENARIONET_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SCENARIONET_DIR}")"
  if ! git clone "${SCENARIONET_REMOTE}" "${SCENARIONET_DIR}"; then
    rm -rf "${SCENARIONET_DIR}"
    git clone "${SCENARIONET_FALLBACK_SOURCE}" "${SCENARIONET_DIR}"
  fi
fi

python -m pip install -e "${SCENARIONET_DIR}"

WAYMO_TOOLKIT_INSTALL_STATUS="ok"
set +e
python -m pip install waymo-open-dataset-tf-2-12-0==1.6.7
WAYMO_TOOLKIT_RC=$?
set -e
if [[ ${WAYMO_TOOLKIT_RC} -ne 0 ]]; then
  WAYMO_TOOLKIT_INSTALL_STATUS="failed"
fi

UNITRAJ_INSTALL_STATUS="ok"
set +e
python -m pip install -e "${UNITRAJ_DIR}" --no-build-isolation
UNITRAJ_RC=$?
set -e
if [[ ${UNITRAJ_RC} -ne 0 ]]; then
  UNITRAJ_INSTALL_STATUS="failed"
fi

python - <<PY > "${ARTIFACT_DIR}/env_summary.txt"
import json
import os
import sys

summary = {
    "env_name": "${ENV_NAME}",
    "python": sys.version,
    "waymo_toolkit_install_status": "${WAYMO_TOOLKIT_INSTALL_STATUS}",
    "unitraj_install_status": "${UNITRAJ_INSTALL_STATUS}",
}

try:
    import torch
    summary["torch"] = torch.__version__
    summary["cuda_available"] = torch.cuda.is_available()
    summary["cuda_version"] = torch.version.cuda
    summary["gpu_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        summary["gpu_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    summary["torch_error"] = repr(exc)

print(json.dumps(summary, indent=2))
PY

echo "${WAYMO_TOOLKIT_INSTALL_STATUS}" > "${ARTIFACT_DIR}/waymo_toolkit_install_status.txt"
echo "${UNITRAJ_INSTALL_STATUS}" > "${ARTIFACT_DIR}/unitraj_install_status.txt"
echo "Environment ready: ${ENV_NAME}"
