#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"
PYTHON=${PYTHON:-${TTT_VENV}/bin/python}

# Exact released-artifact replay defaults to one worker because the randomized
# solver has an empty-rectPool SIGFPE path that high concurrency amplifies.
exec "$PYTHON" "${LOCAL_DIR}/evaluate_ahc039_released_public.py" \
  --workers "${AHC_WORKERS:-1}" \
  --max-cases "${AHC_MAX_CASES:-150}" \
  "$@"
