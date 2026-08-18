"""Compatibility wrapper for the TTT-Discover released TriMul result.

The released kernel defaults ``config.get('nomask', True)``, while the official
input generator never includes that key.  Always taking the masked branch is
mathematically equivalent for all-one masks and fixes random-mask cases without
changing the Triton kernels or numerical computation.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RELEASED = Path(
    os.environ.get("TTT_OFFICIAL_ROOT", _REPO_ROOT.parent / "ttt-discover-official")
) / "results/kernel-engineering/trimul.py"
_SPEC = importlib.util.spec_from_file_location("ttt_released_trimul", _RELEASED)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load released TriMul implementation: {_RELEASED}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def custom_kernel(data):
    inp, mask, weights, config = data
    corrected_config = dict(config)
    corrected_config["nomask"] = False
    return _MODULE.custom_kernel((inp, mask, weights, corrected_config))
