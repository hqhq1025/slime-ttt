"""Entry point for full-parameter TTT-Discover on slime.

This is a thin wrapper around slime's own ``train()`` that additionally
registers the ``--ttt-*`` arguments (via ``parse_args``'s custom-arguments hook).
Run it exactly like ``train.py`` but with ``--rollout-function-path`` pointing at
the TTT rollout and ``--advantage-estimator entropic_adaptive_beta``.

    python3 examples/ttt_discover/train_ttt.py --ttt-env-path ... <other slime args>
"""

from __future__ import annotations

import os
import sys

# Make the slime repo root importable so ``from train import train`` works even
# when this script is launched by path from the examples directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime.utils.arguments import parse_args  # noqa: E402

from examples.ttt_discover.ttt_args import add_ttt_arguments  # noqa: E402

if __name__ == "__main__":
    from train import train  # noqa: E402  (repo-root module)

    args = parse_args(add_ttt_arguments)
    train(args)
