"""TTT-Discover-specific CLI arguments.

These are registered via slime's ``parse_args(add_custom_arguments=...)`` hook
(see ``train_ttt.py``) so the discovery plugin stays self-contained instead of
bloating ``slime/utils/arguments.py``. The general-purpose entropic advantage
estimator lives in slime core; everything here is about *what problem* to solve
and *how the discovery archive behaves*.
"""

from __future__ import annotations

import os
from pathlib import Path


def _official_root() -> str:
    configured = os.environ.get("TTT_OFFICIAL_ROOT")
    if configured:
        return configured
    return str(Path(__file__).resolve().parents[3].parent / "ttt-discover-official")


def _ahc_cache_dir() -> str:
    configured = os.environ.get("TTT_AHC_CACHE_DIR")
    if configured:
        return configured
    return str(Path(__file__).resolve().parents[3].parent / "ttt-storage/ahc-cache/extracted/cache")


def add_ttt_arguments(parser):
    group = parser.add_argument_group("ttt_discover")
    group.add_argument(
        "--ttt-env-path",
        type=str,
        required=True,
        help=(
            "Import path of the TTTEnvironment subclass, e.g. "
            "examples.ttt_discover.envs.erdos.ErdosMinOverlapEnv (':' before the "
            "class name is also accepted)."
        ),
    )
    group.add_argument(
        "--ttt-archive-dir",
        type=str,
        default=None,
        help="Where to persist the discovery archive. Defaults to <save>/ttt_archive.",
    )
    group.add_argument(
        "--ttt-eval-timeout",
        type=int,
        default=600,
        help="Wall-clock seconds allowed for each candidate program in the sandbox.",
    )
    group.add_argument(
        "--ttt-num-cpus-per-task",
        type=int,
        default=1,
        help="CPU thread cap (BLAS) exposed to each sandboxed candidate program.",
    )
    group.add_argument(
        "--ttt-eval-concurrency",
        type=int,
        default=64,
        help="Max number of candidate programs evaluated in parallel (thread pool).",
    )
    group.add_argument(
        "--ttt-max-buffer-size",
        type=int,
        default=1000,
        help="Maximum number of states kept in the discovery archive.",
    )
    group.add_argument("--ttt-puct-c", type=float, default=1.0, help="PUCT exploration constant.")
    group.add_argument(
        "--ttt-topk-children",
        type=int,
        default=2,
        help="Keep at most this many best children per parent when expanding the archive.",
    )
    group.add_argument(
        "--ttt-fail-reward",
        type=float,
        default=0.0,
        help="Reward assigned to invalid / failed candidate programs.",
    )
    group.add_argument(
        "--ttt-target",
        type=float,
        default=None,
        help="Optional override of the environment's target metric shown in prompts.",
    )
    group.add_argument(
        "--ttt-phase1-max-context",
        type=int,
        default=0,
        help=(
            "Prompt + phase-1 token budget for two-phase generation. At the limit, "
            "force the model out of reasoning and reserve the remaining response budget "
            "for final code. Zero disables two-phase generation."
        ),
    )
    group.add_argument(
        "--ttt-context-window",
        type=int,
        default=32768,
        help="Model context window used to cap phase-2 continuation.",
    )
    group.add_argument(
        "--ttt-context-buffer",
        type=int,
        default=50,
        help="Tokens left unused at the end of the model context window.",
    )
    group.add_argument(
        "--ttt-phase2-prefill",
        type=str,
        default=None,
        help=(
            "Optional forced text between reasoning and final generation. By default, "
            "gpt-oss uses the paper's final-channel marker and Qwen uses </think>."
        ),
    )

    # --- Local GPU Mode / TriMul evaluator ---
    group.add_argument(
        "--ttt-trimul-profile",
        choices=["smoke", "full"],
        default="smoke",
        help="TriMul evaluator suite: official-case subset for smoke or all 18 tests + 7 benchmarks.",
    )
    group.add_argument(
        "--ttt-trimul-repeats",
        type=int,
        default=5,
        help="CUDA-event timing repeats per TriMul benchmark (minimum three).",
    )
    group.add_argument(
        "--ttt-trimul-gpu-device",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value used by each isolated TriMul evaluator subprocess.",
    )

    # --- Local GPU Mode / MLA Decode A100 adaptation ---
    group.add_argument(
        "--ttt-mla-repeats",
        type=int,
        default=3,
        help="CUDA-event timing repeats per MLA Decode A100 smoke shape (minimum three).",
    )
    group.add_argument(
        "--ttt-mla-gpu-device",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value used by the isolated MLA Decode evaluator.",
    )
    group.add_argument(
        "--ttt-mla-official-root",
        type=str,
        default=_official_root(),
        help="Official TTT-Discover checkout supplying MLA Decode inputs and reference.",
    )

    # --- Frontier-CS env + remote judge ---
    group.add_argument(
        "--ttt-frontiercs-problems-dir",
        type=str,
        default=None,
        help="Path to Frontier-CS/algorithmic/problems (required for the frontiercs env).",
    )
    group.add_argument(
        "--ttt-frontiercs-problem-id",
        type=str,
        default="0",
        help="Which Frontier-CS problem id to run test-time discovery on.",
    )
    group.add_argument(
        "--ttt-frontiercs-max-score",
        type=float,
        default=100.0,
        help="Max judge score for the problem (used to normalize reward and show target).",
    )
    group.add_argument(
        "--ttt-frontiercs-reward-mode",
        type=str,
        choices=["score", "sum_value", "mean_value"],
        default="score",
        help="How the remote judge result becomes the reward. 'score' (DEFAULT, original: "
        "normalized judge points) | 'sum_value' (sum of each case's 'Value:' parsed from "
        "cases[].msg — a dense, nonzero signal even when every case scores 0 points; for "
        "objective problems like 159) | 'mean_value' (sum_value / num_cases).",
    )
    group.add_argument(
        "--ttt-frontiercs-value-scale",
        type=float,
        default=1.0,
        help="Multiplier on the sum_value/mean_value reward. The entropic advantage is "
        "scale-invariant, so this only changes the logged magnitude (e.g. 1e-6 to keep "
        "best_raw readable).",
    )
    group.add_argument(
        "--ttt-judge-backend",
        type=str,
        choices=["local", "remote"],
        default="local",
        help="Reward backend: 'local' compiles+runs public tests; 'remote' POSTs to the judge.",
    )
    group.add_argument(
        "--ttt-judge-url",
        type=str,
        default=None,
        help="Remote judge endpoint (required when --ttt-judge-backend remote).",
    )
    group.add_argument(
        "--ttt-judge-max-cases",
        type=int,
        default=1,
        help="Local backend: number of public test cases to run per candidate (smoke).",
    )
    group.add_argument(
        "--ttt-judge-timeout",
        type=float,
        default=300.0,
        help="Per-candidate judge timeout in seconds.",
    )
    group.add_argument(
        "--ttt-judge-score-key",
        type=str,
        choices=["score", "scoreUnbounded"],
        default="score",
        help="Remote judge: which field is the reward. 'score' (bounded points) or "
        "'scoreUnbounded' (raw objective; denser signal for optimization problems).",
    )
    group.add_argument(
        "--ttt-judge-poll-interval",
        type=float,
        default=3.0,
        help="Remote judge: seconds between result polls.",
    )

    # --- AtCoder Heuristic Contests (official released ALE-Bench artifacts) ---
    group.add_argument(
        "--ttt-ahc-cache-dir",
        type=str,
        default=_ahc_cache_dir(),
        help="Released ALE-Bench cache containing public inputs and *_tester binaries.",
    )
    group.add_argument(
        "--ttt-ahc-reference-root",
        type=str,
        default=_official_root(),
        help="Official TTT-Discover checkout used for the released AHC prompt and seed code.",
    )
    group.add_argument(
        "--ttt-ahc-max-cases",
        type=int,
        default=1,
        help="Number of released public AHC cases per candidate; 1 is a smoke setting.",
    )
    group.add_argument(
        "--ttt-ahc-time-limit",
        type=float,
        default=2.0,
        help="Official candidate time limit per AHC public case, in seconds.",
    )
    group.add_argument(
        "--ttt-ahc-reward-scale",
        type=float,
        default=1500.0,
        help="Divide mean public score by this value, matching the official AHC evaluator.",
    )
    return parser
