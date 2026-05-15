#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITRAJ_ROOT="$REPO_ROOT/third_party/UniTraj"

if [[ ! -d "$UNITRAJ_ROOT" ]]; then
  echo "Missing UniTraj checkout at $UNITRAJ_ROOT" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the target conda env before running this script." >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_tailrisk_build}"

echo "Using Python: $(which python)"
python - <<'PY'
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY

pushd "$UNITRAJ_ROOT" >/dev/null

rm -rf build
find unitraj/models/mtr/ops -name '*.so' -delete
find unitraj/models/mtr/ops -name '*.o' -delete
find unitraj/models/mtr/ops -name '*.obj' -delete

python setup.py build_ext --inplace

popd >/dev/null

echo "Rebuilt UniTraj MTR ops in place."
