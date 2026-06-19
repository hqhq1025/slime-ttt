"""TTT-Discover rollout for slime (the ``--rollout-function-path``).

This replaces slime's default dataset-driven rollout with the test-time
*discovery loop* from "Learning to Discover at Test Time" (arXiv:2601.16175),
running on slime's full-parameter Megatron + SGLang stack instead of Tinker LoRA.

Each rollout step (= one slime ``rollout_id``):

1. Sample ``rollout_batch_size`` parent states from the discovery archive
   (PUCT tree search over solutions found so far).
2. For each parent, build a prompt conditioned on its best-so-far solution and
   create a group of ``n_samples_per_prompt`` candidate generations.
3. Generate all candidates through the SGLang router (reusing slime's
   token-level ``generate``), capturing rollout log-probs and loss masks.
4. Sandbox-evaluate every candidate to a scalar reward + raw score.
5. Expand the archive with the newly discovered valid solutions.
6. Return the groups; slime then computes **entropic** advantages
   (``--advantage-estimator entropic_adaptive_beta``) and runs a full-parameter
   policy-gradient update, after which the new weights are synced to SGLang.

The single-problem dataset, group sizing, and best-so-far conditioning are the
algorithmic core; full-parameter training is what slime contributes. The
reference's two-phase reasoning sampler is approximated here by single-pass
generation governed by the model's chat template (see the guide for notes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, generate
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

from .archive import DiscoveryArchive
from .environment import TTTEnvironment

logger = logging.getLogger(__name__)

# One controller per RolloutManager actor process; lazily built on first call.
_CONTROLLER: "TTTController | None" = None


def _load_env_class(path: str):
    """Load a TTTEnvironment subclass from ``module.sub.ClassName`` (``:`` also accepted)."""
    # slime's load_function only understands fully-dotted paths; accept the
    # common ``module:Class`` form too by normalizing the colon to a dot.
    cls = load_function(path.replace(":", "."))
    if not (isinstance(cls, type) and issubclass(cls, TTTEnvironment)):
        raise TypeError(f"--ttt-env-path {path!r} must point to a TTTEnvironment subclass")
    return cls


class TTTController:
    def __init__(self, args):
        self.args = args
        env_cls = _load_env_class(args.ttt_env_path)
        self.env: TTTEnvironment = env_cls(args)
        if args.ttt_target is not None:
            self.env.target = args.ttt_target

        archive_dir = args.ttt_archive_dir or os.path.join(args.save or "./", "ttt_archive")
        self.archive = DiscoveryArchive(
            archive_dir,
            batch_size=args.rollout_batch_size,
            create_initial_states=self.env.create_initial_states,
            max_buffer_size=args.ttt_max_buffer_size,
            puct_c=args.ttt_puct_c,
            topk_children=args.ttt_topk_children,
        )
        self.best_dir = os.path.join(args.save or "./", "ttt_best")
        os.makedirs(self.best_dir, exist_ok=True)
        self._eval_pool = ThreadPoolExecutor(max_workers=max(1, args.ttt_eval_concurrency))
        self._sample_index = 0
        self._gen_state = GenerateState(args)  # tokenizer + sampling params + semaphore

    # ---- resume -------------------------------------------------------
    def resume_to(self, rollout_id: int) -> None:
        """Load the most recent archive snapshot at or before ``rollout_id``."""
        for step in range(rollout_id, -1, -1):
            if self.archive.load_from_step(step):
                logger.info(f"[TTT] resumed discovery archive from step {step}")
                return
        self.archive.maybe_seed()

    # ---- prompt building ----------------------------------------------
    def build_sample(self, prompt_text: str, group_index: int) -> Sample:
        tok = self._gen_state.tokenizer
        if getattr(self.args, "apply_chat_template", False) and hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = prompt_text
        s = Sample(prompt=prompt, group_index=group_index, index=self._sample_index)
        self._sample_index += 1
        return s

    # ---- best-so-far logging ------------------------------------------
    def dump_best(self, rollout_id: int) -> dict:
        best = self.archive.best_state()
        if best is None:
            return {}
        raw = best.value if self.env.maximize else -best.value
        payload = {
            "rollout_id": rollout_id,
            "raw_score": raw,
            "value": best.value,
            "code": best.code,
        }
        with open(os.path.join(self.best_dir, "best.json"), "w") as f:
            json.dump(payload, f, indent=2)
        return {"ttt/best_raw_score": float(raw)}


def _get_controller(args) -> TTTController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = TTTController(args)
    return _CONTROLLER


async def _generate_rollout_async(args, rollout_id: int) -> RolloutFnTrainOutput:
    ctrl = _get_controller(args)
    if rollout_id == args.start_rollout_id:
        ctrl.resume_to(rollout_id)

    env = ctrl.env
    n = args.n_samples_per_prompt
    sampling_params = ctrl._gen_state.sampling_params.copy()

    # 1) sample parents and 2) build candidate groups
    parents = ctrl.archive.sample(args.rollout_batch_size)
    groups: list[list[Sample]] = []
    group_parent: list = []
    group_prompts: list[str] = []
    for gi, parent in enumerate(parents):
        question = env.build_prompt(parent)
        group = [ctrl.build_sample(question, group_index=gi) for _ in range(n)]
        groups.append(group)
        group_parent.append(parent)
        group_prompts.append(question)

    flat = [s for g in groups for s in g]

    # 3) generate everything through SGLang (bounded by the engine semaphore)
    sem = ctrl._gen_state.semaphore

    async def _gen(sample: Sample):
        async with sem:
            return await generate(args, sample, sampling_params.copy())

    flat = list(await asyncio.gather(*[_gen(s) for s in flat]))

    # regroup (order preserved by gather)
    regrouped: list[list[Sample]] = [flat[i * n : (i + 1) * n] for i in range(len(groups))]

    # 4) sandbox-evaluate every candidate (thread pool; subprocesses release GIL)
    loop = asyncio.get_running_loop()

    async def _score(sample: Sample, parent):
        result = await loop.run_in_executor(ctrl._eval_pool, env.evaluate, sample.response, parent)
        sample.reward = float(result.reward)
        sample.label = f"{result.raw_score:.6f}"
        sample.metadata = {
            **(sample.metadata or {}),
            "raw_score": float(result.raw_score),
            "correctness": float(result.correctness),
            "parent_id": parent.id,
            "msg": result.msg,
        }
        return result

    score_tasks = []
    for parent, group in zip(group_parent, regrouped):
        for sample in group:
            score_tasks.append(_score(sample, parent))
    results = list(await asyncio.gather(*score_tasks))

    # 5) expand the archive with valid children (best-per-parent handled inside)
    children, child_parents = [], []
    ri = 0
    for parent, group in zip(group_parent, regrouped):
        for sample in group:
            result = results[ri]
            ri += 1
            if result.correctness > 0:
                code = env.extract_code(sample.response) or ""
                children.append(env.make_child_state(rollout_id, code, result))
                child_parents.append(parent)
    n_added = ctrl.archive.update(children, child_parents, step=rollout_id)

    # 6) metrics
    rewards = [s.reward for s in flat]
    raw_scores = [s.metadata.get("raw_score", 0.0) for s in flat]
    n_correct = sum(1 for s in flat if s.metadata.get("correctness", 0.0) > 0)
    metrics = {
        "ttt/frac_correct": n_correct / max(1, len(flat)),
        "ttt/reward_mean": sum(rewards) / max(1, len(rewards)),
        "ttt/reward_max": max(rewards) if rewards else 0.0,
        "ttt/raw_score_max": max(raw_scores) if raw_scores else 0.0,
        "ttt/children_added": n_added,
        "ttt/num_parents": len(parents),
    }
    metrics.update(ctrl.archive.stats())
    metrics.update(ctrl.dump_best(rollout_id))

    logger.info(
        f"[TTT] step {rollout_id}: {n_correct}/{len(flat)} valid, "
        f"best_raw={metrics.get('ttt/best_raw_score')}, archive={metrics.get('ttt/archive_size')}"
    )
    return RolloutFnTrainOutput(samples=regrouped, metrics=metrics)


def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    """slime rollout entry point. ``data_source`` is unused (states come from the archive)."""
    if evaluation:
        # The discovery archive *is* the evaluation signal; we expose no separate
        # eval dataset. Return an empty eval output so --eval-interval is a no-op.
        return RolloutFnEvalOutput(data={})
    return run(_generate_rollout_async(args, rollout_id))
