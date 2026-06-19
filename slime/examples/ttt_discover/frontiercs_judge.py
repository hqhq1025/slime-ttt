"""Scoring backends for Frontier-CS C++ solutions.

Two backends, selected by ``--ttt-judge-backend``:

* ``remote`` (recommended for training): POST the candidate to a judge server
  that compiles, runs the full (private) test suite, and returns a **points**
  score. This is the path the user runs on a separate machine. The exact request
  /response schema is confirmed with the judge owner; the default below is a
  reasonable JSON contract and is the single place to adjust once finalized.

* ``local`` (smoke / offline debugging): compile with g++ and run the available
  public ``testdata`` cases through the problem's testlib checker
  (``chk.cc``/``checker.cpp``). Returns the fraction of cases the checker accepts
  (×100). It does **not** reproduce partial/competitive scoring — use ``remote``
  for a real training signal.

Higher score = better (judge points), regardless of whether the underlying
objective is a min or max — the checker encodes the direction.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JudgeResult:
    score: float           # higher = better (judge points); 0.0 on failure
    max_score: float       # scale (e.g. 100.0)
    status: str            # ok | compile_error | runtime_error | tle | wa | error
    msg: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        return self.score / self.max_score if self.max_score else 0.0


# --------------------------------------------------------------------------- #
# Remote judge (TODO: confirm exact schema with the judge owner)
# --------------------------------------------------------------------------- #
def _normalize_base(judge_url: str) -> str:
    base = judge_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    return base


def remote_judge(
    problem_id: str,
    code: str,
    judge_url: str,
    *,
    timeout: float = 300.0,
    poll_interval: float = 3.0,
    score_key: str = "score",
    lang: str = "cpp",
    max_score: float = 100.0,
) -> JudgeResult:
    """Submit to the async judge at yanagiorigami.uk and poll for the result.

    Protocol (discovered against the live judge):
        POST  {base}/submit   {"pid": str, "lang": "cpp", "code": str}  -> {"sid": int}
        GET   {base}/result/{sid}  -> {"status": "done"|..., "passed": bool,
              "result": str, "score": float, "scoreUnbounded": float, "cases": [...]}

    ``judge_url`` is the bare base (e.g. ``https://yanagiorigami.uk``); the
    ``/submit`` and ``/result`` routes are appended here. ``score_key`` selects
    which field becomes the reward: ``score`` (bounded competition points, the
    default) or ``scoreUnbounded`` (raw objective — a denser signal for
    optimization problems where many candidates score 0).
    """
    base = _normalize_base(judge_url)
    # A browser-like UA: Cloudflare 403s the default "Python-urllib" agent.
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ttt-discover-judge/1.0",
          "Accept": "application/json"}

    # 1) submit
    payload = json.dumps({"pid": str(problem_id), "lang": lang, "code": code}).encode()
    try:
        req = urllib.request.Request(
            base + "/submit", data=payload,
            headers={"Content-Type": "application/json", **ua}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            sub = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        return JudgeResult(0.0, max_score, "error", msg=f"submit failed: {e}")
    sid = sub.get("sid")
    if sid is None:
        return JudgeResult(0.0, max_score, "error", msg=f"no sid in response: {sub}")

    # 2) poll for completion
    result_url = f"{base}/result/{sid}"
    deadline = time.time() + timeout
    data: dict = {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(result_url, headers=ua), timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            time.sleep(poll_interval)
            continue
        if str(data.get("status")) == "done":
            break
        time.sleep(poll_interval)
    else:
        return JudgeResult(0.0, max_score, "timeout", msg=f"judge timeout sid={sid}", detail={"sid": sid})

    score = float(data.get(score_key, data.get("score", 0.0)) or 0.0)
    passed = bool(data.get("passed", False))
    result = str(data.get("result", "unknown"))
    # "score" is out of max_score; scoreUnbounded is raw (no normalization).
    ms = max_score if score_key == "score" else 1.0
    status = "ok" if passed else result
    return JudgeResult(
        score=score, max_score=ms, status=status, msg=result,
        detail={"sid": sid, "passed": passed, "result": result,
                "score": data.get("score"), "scoreUnbounded": data.get("scoreUnbounded")},
    )


# --------------------------------------------------------------------------- #
# Local judge (compile + run public tests + testlib checker)
# --------------------------------------------------------------------------- #
_GPP_FLAGS = ["-std=c++17", "-O2", "-pipe", "-DNDEBUG"]


def _run(cmd, stdin=None, timeout=None, cwd=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, cwd=cwd, check=False)
        return p.returncode, p.stdout, p.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or b"", e.stderr or b"", time.time() - t0


def _time_limit_s(problem_dir: Path, pad: int) -> int:
    try:
        m = re.search(r"time:\s*(\d+)s", (problem_dir / "config.yaml").read_text())
        return (int(m.group(1)) + pad) if m else (5 + pad)
    except Exception:
        return 5 + pad


def local_judge(
    problem_id: str,
    code: str,
    problems_dir: str,
    *,
    time_pad: int = 2,
    max_cases: int = 1,
) -> JudgeResult:
    pdir = Path(problems_dir) / str(problem_id)
    if not (pdir / "statement.txt").exists():
        return JudgeResult(0.0, 100.0, "error", msg=f"problem {problem_id} not found in {problems_dir}")

    with tempfile.TemporaryDirectory(prefix="fcs_") as tmp:
        tmp = Path(tmp)
        src = tmp / "sol.cpp"
        src.write_text(code)
        sol = tmp / "sol"
        rc, _, err, _ = _run(["g++", *_GPP_FLAGS, str(src), "-o", str(sol)], timeout=120)
        if rc != 0:
            return JudgeResult(0.0, 100.0, "compile_error", msg=err.decode("utf-8", "replace")[:1500])

        # Build testlib checker if present.
        chk_src = pdir / "chk.cc"
        if not chk_src.exists():
            chk_src = pdir / "checker.cpp"
        chk_bin = None
        if chk_src.exists():
            testlib = pdir.parent.parent / "judge" / "testlib.h"
            if not testlib.exists():
                found = list((pdir.parent.parent).rglob("testlib.h"))
                testlib = found[0] if found else None
            chk_bin = tmp / "chk"
            cc = ["g++", "-std=c++17", "-O2", str(chk_src), "-o", str(chk_bin)]
            if testlib:
                cc += ["-I", str(testlib.parent)]
            if _run(cc, timeout=120)[0] != 0:
                chk_bin = None  # fall back to diff

        td = pdir / "testdata"
        ins = sorted(td.glob("*.in"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1_000_000)
        ins = ins[:max_cases] if max_cases > 0 else ins
        if not ins:
            return JudgeResult(0.0, 100.0, "no_public_test")

        run_to = _time_limit_s(pdir, time_pad)
        passed = 0
        last = ""
        for in_file in ins:
            ans_file = in_file.with_suffix(".ans")
            rc_run, out_bytes, err_run, _ = _run([str(sol)], stdin=in_file.read_bytes(), timeout=run_to)
            if rc_run == 124:
                last = "tle"
                continue
            if rc_run != 0:
                last = "runtime_error"
                continue
            out_file = tmp / "out"
            out_file.write_bytes(out_bytes)
            if chk_bin is not None and ans_file.exists():
                rc_chk, ck_out, ck_err, _ = _run(
                    [str(chk_bin), str(in_file), str(out_file), str(ans_file)], timeout=30)
                ok = rc_chk == 0
                last = (ck_out + ck_err).decode("utf-8", "replace")[:200]
            elif ans_file.exists():
                ok = out_bytes.split() == ans_file.read_text().split()
                last = "diff"
            else:
                ok = rc_run == 0
            passed += int(ok)

        frac = passed / len(ins)
        status = "ok" if passed == len(ins) else ("wa" if last not in ("tle", "runtime_error") else last)
        return JudgeResult(score=100.0 * frac, max_score=100.0, status=status,
                           msg=f"{passed}/{len(ins)} public cases", detail={"last": last})


def score_solution(problem_id, code, *, backend, problems_dir=None, judge_url=None,
                   timeout=300.0, max_cases=1, score_key="score", poll_interval=3.0,
                   max_score=100.0) -> JudgeResult:
    if backend == "remote":
        if not judge_url:
            return JudgeResult(0.0, max_score, "error", msg="remote backend requires --ttt-judge-url")
        return remote_judge(problem_id, code, judge_url, timeout=timeout,
                            poll_interval=poll_interval, score_key=score_key, max_score=max_score)
    return local_judge(problem_id, code, problems_dir, max_cases=max_cases)
