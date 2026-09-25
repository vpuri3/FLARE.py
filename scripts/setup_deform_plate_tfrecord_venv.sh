#!/usr/bin/env bash
# Isolated venv for deform_plate TFRecord → NPZ conversion (tfrecord package only; no TensorFlow).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/out/pdebench/.venv-deform-plate-tfrecord"
REQ="${ROOT}/out/pdebench/deform_plate_tfrecord/requirements.txt"

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "Main .venv missing. Run scripts/install.sh first." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[deform_plate] creating tfrecord sub-env at ${VENV_DIR}"
  "${ROOT}/.venv/bin/python" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install -q --upgrade pip
"${VENV_DIR}/bin/pip" install -q -r "${REQ}"
echo "[deform_plate] tfrecord sub-env ready: ${VENV_DIR}/bin/python"
