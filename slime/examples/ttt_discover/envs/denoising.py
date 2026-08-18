"""Local slime environment for the official pancreas denoising evaluator."""

from __future__ import annotations

import ast
import json
import math
import os
import resource
import subprocess
import tempfile
from pathlib import Path

from ..environment import RewardResult, TTTEnvironment
from ..state import State


def _extract_seed(path: Path) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "magic_denoise":
            segment = ast.get_source_segment(source, node)
            if segment:
                return segment
    raise RuntimeError(f"magic_denoise not found in {path}")


def _limit_worker() -> None:
    # The released artifact peaks near 6.1 GB.  Leave enough headroom for
    # generated NumPy intermediates while preventing a bad candidate from
    # consuming the full 755-GB host.
    resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class DenoisingPancreasEnv(TTTEnvironment):
    name = "denoising_pancreas"
    metric_name = "pancreas MSE"
    entrypoint = "magic_denoise"
    maximize = False
    target = 0.15

    def __init__(self, args):
        super().__init__(args)
        repo_root = Path(__file__).resolve().parents[4]
        storage_root = Path(os.environ.get("TTT_STORAGE_ROOT", repo_root.parent / "ttt-storage"))
        self.storage_root = storage_root
        self.official_root = Path(os.environ.get(
            "TTT_DENOISING_OFFICIAL_ROOT",
            os.environ.get("TTT_OFFICIAL_ROOT", repo_root.parent / "ttt-discover-official"),
        ))
        self.python = Path(os.environ.get(
            "TTT_DENOISING_PYTHON", storage_root / "denoising-venv/bin/python"
        ))
        self.worker = Path(__file__).resolve().parents[4] / "local/evaluate_denoising.py"
        self.dataset = Path(os.environ.get(
            "OPENPROBLEMS_PANCREAS_PATH",
            storage_root / "datasets/openproblems/pancreas.h5ad",
        ))
        self.cache = Path(os.environ.get(
            "OPENPROBLEMS_CACHE_DIR", storage_root / "openproblems_cache"
        ))
        required = [self.python, self.worker, self.dataset]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing denoising runtime files: " + ", ".join(missing))

        seed_path = self.official_root / "examples/denoising/utils.py"
        self.seed_code = _extract_seed(seed_path)
        self.seed_mse = 0.23141190791839888
        self.seed_poisson = 0.03692228008772968

    def create_initial_states(self, n: int) -> list[State]:
        return [
            State(
                timestep=-1,
                code=self.seed_code,
                construction={"mse": self.seed_mse, "poisson": self.seed_poisson},
                value=-self.seed_mse,
            )
            for _ in range(n)
        ]

    def build_prompt(self, state: State) -> str:
        context = state.to_prompt(
            self.target,
            metric_name=self.metric_name,
            maximize=False,
            language="python",
        )
        return f"""You are an expert in computational biology and single-cell RNA-seq denoising.

Implement a complete `magic_denoise(X, **kwargs)` function. `X` is a raw non-negative count matrix with shape (cells, genes); return a finite, non-negative NumPy array with the same shape. Available libraries include NumPy, SciPy, scikit-learn, scanpy, scprep, anndata, and graphtools. Do not use filesystem, network, subprocess, or GPU operations.

Evaluation uses the official OpenProblems pancreas/inDrop1 molecular split at seed 42. Lower log-normalized MSE is better. Poisson loss is a hard validity constraint:
`poisson_norm = (0.257575 - poisson) / (0.257575 - 0.031739)` must be at least 0.97.
The reward after passing that constraint is `1 / MSE`. The evaluator timeout is {self.eval_timeout} seconds and candidates receive two CPU threads.

{context}

Return all imports, helper functions, and `magic_denoise` in one final ```python code block. Preserve the function signature and produce a complete replacement, not a patch.
"""

    def evaluate(self, response_text: str, state: State) -> RewardResult:
        del state
        code = self.extract_code(response_text)
        if not code:
            return self.failure("no python code block found")

        with tempfile.TemporaryDirectory(prefix="ttt_denoising_") as tmp_name:
            tmp = Path(tmp_name)
            candidate = tmp / "candidate.py"
            output = tmp / "result.json"
            candidate.write_text(code)
            env = os.environ.copy()
            env.update({
                # Do not retain slime's regular ``examples`` package here: it
                # masks the official checkout's namespace package.
                "PYTHONPATH": str(self.official_root),
                "OPENPROBLEMS_CACHE_DIR": str(self.cache),
                "OPENPROBLEMS_PANCREAS_PATH": str(self.dataset),
                "NUMBA_CACHE_DIR": str(self.storage_root / "numba_cache"),
                "TMPDIR": str(self.storage_root / "tmp/denoising"),
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            })
            cmd = [
                str(self.python),
                str(self.worker),
                str(candidate),
                "--seed", "42",
                "--output", str(output),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.eval_timeout,
                    env=env,
                    start_new_session=True,
                    preexec_fn=_limit_worker,
                )
            except subprocess.TimeoutExpired as exc:
                return self.failure(
                    f"denoising evaluation timed out after {self.eval_timeout}s",
                    (exc.stdout or "")[-8000:],
                )
            stdout = proc.stdout[-8000:]
            if proc.returncode != 0 or not output.exists():
                return self.failure(
                    f"denoising evaluator exited {proc.returncode} without a result",
                    stdout,
                )
            try:
                result = json.loads(output.read_text())
            except Exception as exc:  # noqa: BLE001
                return self.failure(f"invalid denoising result: {exc}", stdout)

        mse = float(result["mse"])
        poisson = float(result["poisson"])
        if not math.isfinite(mse) or not math.isfinite(poisson) or not result.get("valid"):
            return self.failure(
                f"Poisson constraint failed: MSE={mse:.6f}, Poisson={poisson:.6f}",
                stdout,
            )
        return RewardResult(
            reward=1.0 / mse,
            raw_score=mse,
            correctness=1.0,
            construction={"mse": mse, "poisson": poisson},
            msg=f"MSE {mse:.6f}; Poisson {poisson:.6f}; valid",
            stdout=stdout,
            metrics=result,
        )
