#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"
PYTHON=${PYTHON:-${TTT_VENV}/bin/python}

exec "$PYTHON" "${LOCAL_DIR}/evaluate_ahc058_released_public.py" \
  --workers "${AHC_WORKERS:-64}" \
  --max-cases "${AHC_MAX_CASES:-150}" \
  "$@"
