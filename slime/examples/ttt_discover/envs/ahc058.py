"""Local AHC058 environment aligned with the released TTT-Discover task.

The official experiment evaluates C++20 programs on 150 ALE-Bench inputs and
uses ``mean(raw AtCoder score) / 3_000_000`` as its training reward.  This
module provides a dependency-free public-input backend for smoke runs.  Its
parser, state transition, invalid-action handling, and score formula mirror the
released Rust verifier in ``examples/ahc/lib/problems/ahc058/tools/src/lib.rs``.

This is intentionally a local reproduction evaluator, not a hardened sandbox:
candidate C++ is compiled and run as the current user with CPU, address-space,
and output-size limits.
"""

from __future__ import annotations

import math
import os
import re
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..ahc_judge import judge_ahc_public
from ..environment import RewardResult, TTTEnvironment
from ..state import State


_FORMAT = """Use C++20. Read one instance from stdin and print exactly 500 actions.
For an upgrade print `level id`; to do nothing print `-1`. Comment lines may
start with `#`. Put the complete program in one final ```cpp code block and do
not include prose after it."""


def _extract_cpp(text: str) -> str | None:
    matches = re.findall(r"```(?:cpp|c\+\+|cxx)\s*\n(.*?)(?:\n```|$)", text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    code = matches[-1].strip()
    return code or None


@dataclass
class AHC058CaseResult:
    input_file: str
    score: float = 0.0
    status: str = "error"
    message: str = ""
    elapsed_s: float = 0.0


@dataclass
class AHC058JudgeResult:
    cases: list[AHC058CaseResult] = field(default_factory=list)
    compile_error: str = ""

    @property
    def accepted(self) -> int:
        return sum(case.status == "ok" for case in self.cases)

    @property
    def mean_score(self) -> float:
        # The released environment averages over all selected cases; failures
        # contribute zero rather than disappearing from the denominator.
        return sum(case.score for case in self.cases) / len(self.cases) if self.cases else 0.0


def parse_input(text: str) -> tuple[int, int, int, int, list[int], list[list[int]]]:
    values = [int(x) for x in text.split()]
    if len(values) < 4:
        raise ValueError("input is missing N L T K")
    n, levels, turns, apples = values[:4]
    expected = 4 + n + levels * n
    if len(values) != expected:
        raise ValueError(f"input has {len(values)} integers, expected {expected}")
    a = values[4 : 4 + n]
    flat_c = values[4 + n :]
    costs = [flat_c[i * n : (i + 1) * n] for i in range(levels)]
    return n, levels, turns, apples, a, costs


def score_output(input_text: str, output_text: str) -> tuple[int, str]:
    """Score one output exactly like the official AHC058 Rust verifier."""
    n, levels, turns, apples, production, costs = parse_input(input_text)
    actions: list[tuple[int, int] | None] = []
    for line_number, raw_line in enumerate(output_text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        try:
            level = int(tokens[0])
        except (ValueError, IndexError):
            return 0, f"line {line_number}: invalid level"
        if level == -1:
            if len(tokens) != 1:
                return 0, f"line {line_number}: too many tokens"
            actions.append(None)
        else:
            if not 0 <= level < levels:
                return 0, f"line {line_number}: level out of range"
            if len(tokens) != 2:
                return 0, f"line {line_number}: expected level and id"
            try:
                machine_id = int(tokens[1])
            except ValueError:
                return 0, f"line {line_number}: invalid id"
            if not 0 <= machine_id < n:
                return 0, f"line {line_number}: id out of range"
            actions.append((level, machine_id))
        # The official parser stops after T actions and ignores later lines.
        if len(actions) == turns:
            break

    if len(actions) < turns:
        return 0, f"not enough actions: expected {turns}, got {len(actions)}"

    counts = [[1] * n for _ in range(levels)]
    power = [[0] * n for _ in range(levels)]
    for turn, action in enumerate(actions):
        if action is not None:
            level, machine_id = action
            cost = costs[level][machine_id] * (power[level][machine_id] + 1)
            if apples < cost:
                return 0, f"not enough apples at turn {turn}: have {apples}, need {cost}"
            apples -= cost
            power[level][machine_id] += 1

        # Level order is significant: a higher-level machine affects the next
        # turn because its lower level has already produced on this turn.
        for level in range(levels):
            for machine_id in range(n):
                if level == 0:
                    apples += production[machine_id] * counts[level][machine_id] * power[level][machine_id]
                else:
                    counts[level - 1][machine_id] += counts[level][machine_id] * power[level][machine_id]

    if apples <= 0:
        return 0, f"non-positive apples at the end: {apples}"
    return round(100_000 * math.log2(apples)), ""


def _limit_candidate() -> None:
    # AHC058's official memory limit is 1 GiB. Limit output files as well so a
    # malformed candidate cannot fill the local filesystem during a smoke run.
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def judge_cpp(
    code: str,
    input_files: list[Path],
    *,
    case_timeout: float = 3.0,
) -> AHC058JudgeResult:
    """Compile once, then evaluate a candidate on the selected public inputs."""
    result = AHC058JudgeResult()
    with tempfile.TemporaryDirectory(prefix="ttt_ahc058_") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "main.cpp"
        binary = tmp / "main"
        source.write_text(code)
        try:
            compiled = subprocess.run(
                ["g++", "-std=c++20", "-O2", "-pipe", "-DNDEBUG", str(source), "-o", str(binary)],
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result.compile_error = "compilation timed out"
            return result
        if compiled.returncode != 0:
            result.compile_error = compiled.stderr.decode("utf-8", "replace")[:2000]
            return result

        for input_file in input_files:
            case = AHC058CaseResult(input_file=input_file.name)
            start = time.monotonic()
            try:
                run = subprocess.run(
                    [str(binary)],
                    input=input_file.read_bytes(),
                    cwd=tmp,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=case_timeout,
                    check=False,
                    preexec_fn=_limit_candidate,
                )
            except subprocess.TimeoutExpired:
                case.status = "tle"
                case.message = f"exceeded {case_timeout:.1f}s"
            else:
                if run.returncode != 0:
                    case.status = "runtime_error"
                    case.message = run.stderr.decode("utf-8", "replace")[:500]
                else:
                    score, error = score_output(
                        input_file.read_text(), run.stdout.decode("utf-8", "replace")
                    )
                    case.score = float(score)
                    case.status = "wa" if error else "ok"
                    case.message = error
            case.elapsed_s = time.monotonic() - start
            result.cases.append(case)
    return result


def default_input_dir() -> Path:
    configured = os.environ.get("TTT_AHC058_INPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    # In the portable layout, slime-ttt and the released reference are siblings.
    account_root = Path(__file__).resolve().parents[5]
    return account_root / "ttt-discover-official/examples/ahc/lib/problems/ahc058/tools/in"


class AHC058Env(TTTEnvironment):
    name = "ahc058"
    metric_name = "mean public score"
    code_language = "cpp"
    maximize = True
    target = 6_500_000.0

    def __init__(self, args):
        super().__init__(args)
        default_cache = getattr(
            args,
            "ttt_ahc_cache_dir",
            Path(__file__).resolve().parents[4].parent / "ttt-storage/ahc-cache/extracted/cache",
        )
        self.cache_dir = Path(os.environ.get("TTT_AHC_CACHE_DIR", default_cache))
        self.max_cases = int(os.environ.get(
            "TTT_AHC058_MAX_CASES", str(getattr(args, "ttt_ahc_max_cases", 1))
        ))
        self.case_timeout = float(os.environ.get(
            "TTT_AHC058_CASE_TIMEOUT", str(getattr(args, "ttt_ahc_time_limit", 2.0))
        ))
        default_reference = getattr(
            args,
            "ttt_ahc_reference_root",
            Path(__file__).resolve().parents[4].parent / "ttt-discover-official",
        )
        reference_root = Path(os.environ.get("TTT_OFFICIAL_ROOT", default_reference))
        statement_path = reference_root / "examples/ahc/lib/problems/ahc058/statement_en.md"
        if not statement_path.exists():
            raise FileNotFoundError(f"AHC058 statement not found: {statement_path}")
        self.statement = statement_path.read_text()
        self.warm_start = os.environ.get("TTT_AHC058_WARM_START", "0") == "1"
        self.seed_code = ""
        self.seed_value = 0.0
        if self.warm_start:
            default_seed = Path(__file__).resolve().parents[4] / "local/ahc058_greedy_baseline.cpp"
            seed_path = Path(os.environ.get("TTT_AHC058_SEED", str(default_seed)))
            self.seed_code = seed_path.read_text()
            seed_result = judge_ahc_public(
                self.seed_code,
                problem_id="ahc058",
                cache_dir=self.cache_dir,
                max_cases=self.max_cases,
                time_limit=self.case_timeout,
            )
            if seed_result.status != "ok":
                raise RuntimeError(f"AHC058 warm-start seed failed: {seed_result.msg}")
            self.seed_value = float(seed_result.raw_score)

    def create_initial_states(self, n: int) -> list[State]:
        return [
            State(
                timestep=-1,
                code=self.seed_code,
                construction=None,
                value=self.seed_value,
            )
            for _ in range(n)
        ]

    def extract_code(self, response_text: str) -> str | None:
        return _extract_cpp(response_text)

    def build_prompt(self, state: State) -> str:
        prompt = (
            "You are a world-class algorithm engineer. Solve this heuristic optimization "
            "problem and maximize the score.\n\n"
            f"{self.statement}\n\nRules:\n{_FORMAT}\n"
        )
        if state.code and state.value is not None:
            prompt += (
                "\nImprove the previous solution substantially while preserving validity.\n"
                + state.to_prompt(self.target, metric_name=self.metric_name, maximize=True, language="cpp")
                + "\n"
            )
        return prompt + "\n### Answer:\n```cpp\n// YOUR COMPLETE SOLUTION HERE\n```\n"

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        del state
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no cpp code block found")
        judged = judge_ahc_public(
            code,
            problem_id="ahc058",
            cache_dir=self.cache_dir,
            max_cases=self.max_cases,
            time_limit=self.case_timeout,
        )
        if judged.status != "ok":
            return self.failure(f"{judged.status}: {judged.msg}")
        raw_score = judged.raw_score
        # Exact released AhcRewardEvaluator normalization for AHC058:
        # (mean score / 1500) / 2000.
        reward = raw_score / 3_000_000.0
        message = f"accepted {judged.passed}/{judged.num_cases}, mean score {raw_score:.1f}"
        return RewardResult(
            reward=reward,
            raw_score=raw_score,
            correctness=1.0,
            construction=code,
            msg=message,
            metrics={
                "accepted": judged.passed,
                "num_cases": judged.num_cases,
                "compile_seconds": judged.compile_seconds,
                "run_seconds": judged.run_seconds,
            },
        )
