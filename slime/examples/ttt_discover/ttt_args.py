"""TTT-Discover-specific CLI arguments.

These are registered via slime's ``parse_args(add_custom_arguments=...)`` hook
(see ``train_ttt.py``) so the discovery plugin stays self-contained instead of
bloating ``slime/utils/arguments.py``. The general-purpose entropic advantage
estimator lives in slime core; everything here is about *what problem* to solve
and *how the discovery archive behaves*.
"""

from __future__ import annotations


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
    return parser
