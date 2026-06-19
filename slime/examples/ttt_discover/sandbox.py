"""Sandboxed execution of model-generated programs.

A trimmed, self-contained port of TTT-Discover's ``run_with_timeout``. It writes
the candidate program to a temp file and runs a named entry-point function in a
fresh ``spawn`` subprocess with: filesystem-write blocking, BLAS thread caps, a
hard wall-clock timeout, and process-group SIGKILL cleanup. The entry point's
return value is pickled back; stdout is captured for prompt conditioning.

The reference farmed these out across a Ray CPU pool. Here we run them in the
``RolloutManager`` actor and bound concurrency with a thread pool — adequate for
single-node experiments and dependency-free. For large multi-node sweeps, swap
``run_python_entrypoint`` for a ``ray.remote`` wrapper (see the guide).

Security: generated code is executed. Run only on an isolated machine/VPN.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Injected into the child before the candidate program runs: force spawn, block
# filesystem mutations, silence stdout/stderr inside worker pools.
_HARNESS = r'''
import sys, os, pickle, traceback
import importlib.util as _il

try:
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
except Exception:
    pass

def _block_fs_writes():
    try:
        import builtins
        _orig_open = builtins.open
        def _ro_open(file, mode="r", *a, **k):
            if any(c in mode for c in ("w", "a", "+", "x")):
                raise PermissionError("File writes are disabled in the sandbox")
            return _orig_open(file, mode, *a, **k)
        builtins.open = _ro_open
        def _blocked(*a, **k):
            raise PermissionError("Filesystem mutation disabled in the sandbox")
        for _n in ("remove", "unlink", "rename", "replace", "rmdir", "mkdir",
                   "makedirs", "chmod", "chown", "link", "symlink"):
            if hasattr(os, _n):
                setattr(os, _n, _blocked)
    except Exception:
        pass

_block_fs_writes()

_PROGRAM_PATH = "__PROGRAM_PATH__"
_ENTRYPOINT = "__ENTRYPOINT__"
_RESULTS_PATH = "__RESULTS_PATH__"

sys.path.insert(0, os.path.dirname(_PROGRAM_PATH))
try:
    spec = _il.spec_from_file_location("candidate_program", _PROGRAM_PATH)
    program = _il.module_from_spec(spec)
    spec.loader.exec_module(program)
    func = getattr(program, _ENTRYPOINT)
    result = func()
    with _real_open(_RESULTS_PATH, "wb") as f:
        pickle.dump(result, f)
except Exception as e:
    try:
        with _real_open(_RESULTS_PATH, "wb") as f:
            pickle.dump({"__error__": f"{type(e).__name__}: {e}"}, f)
    except Exception:
        pass
    traceback.print_exc()
'''

# The harness blocks writes, but it must still write its own result file. We
# capture the real open() before blocking and expose it as ``_real_open``.
_REAL_OPEN_SHIM = "import builtins as _b; _real_open = _b.open\n"


@dataclass
class SandboxResult:
    ok: bool
    value: object = None
    error: str = ""
    stdout: str = ""


def run_python_entrypoint(
    code: str,
    entrypoint: str,
    timeout_s: int,
    *,
    preamble: str = "",
    cpus: list[int] | None = None,
) -> SandboxResult:
    """Run ``entrypoint()`` defined in ``preamble + code`` in a sandboxed subprocess.

    Returns a :class:`SandboxResult`. On any failure (timeout, exception, missing
    result) ``ok`` is False and ``error`` describes the cause.
    """
    program_src = (preamble + "\n\n" + code) if preamble else code

    with tempfile.TemporaryDirectory(prefix="ttt_sandbox_") as tmp:
        program_path = os.path.join(tmp, "candidate.py")
        results_path = os.path.join(tmp, "result.pkl")
        runner_path = os.path.join(tmp, "runner.py")
        with open(program_path, "w") as f:
            f.write(program_src)

        runner_src = _REAL_OPEN_SHIM + (
            _HARNESS.replace("__PROGRAM_PATH__", program_path)
            .replace("__ENTRYPOINT__", entrypoint)
            .replace("__RESULTS_PATH__", results_path)
        )
        with open(runner_path, "w") as f:
            f.write(runner_src)

        env = os.environ.copy()
        t = str(max(1, len(cpus) if cpus else 1))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            env.setdefault(var, t)

        stdout_bytes = b""
        try:
            proc = subprocess.Popen(
                [sys.executable, runner_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                stdout_bytes, _ = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                return SandboxResult(ok=False, error=f"timeout after {timeout_s}s",
                                     stdout=_decode(stdout_bytes))
            finally:
                _kill_tree(proc)
        except Exception as e:  # pragma: no cover - launch failure
            return SandboxResult(ok=False, error=f"launch failed: {e}")

        stdout = _decode(stdout_bytes)
        if not os.path.exists(results_path):
            return SandboxResult(ok=False, error="no result produced", stdout=stdout)
        try:
            with open(results_path, "rb") as f:
                value = pickle.load(f)
        except Exception as e:
            return SandboxResult(ok=False, error=f"unpickle failed: {e}", stdout=stdout)
        if isinstance(value, dict) and "__error__" in value:
            return SandboxResult(ok=False, error=str(value["__error__"]), stdout=stdout)
        return SandboxResult(ok=True, value=value, stdout=stdout)


def _kill_tree(proc: subprocess.Popen) -> None:
    import signal

    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
            except Exception:
                pass
        try:
            proc.wait(timeout=0.5)
            break
        except Exception:
            pass


def _decode(b: bytes) -> str:
    try:
        return b.decode(errors="ignore")
    except Exception:
        return ""
