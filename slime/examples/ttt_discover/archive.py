"""Discovery archive — the PUCT tree-search engine over candidate solutions.

This is a trimmed, dependency-light port of TTT-Discover's ``PUCTSampler``. It is
the "search" half of the method: it keeps a population of past solutions
(:class:`~examples.ttt_discover.state.State`) and, each rollout step, selects
which ones to expand next using a PUCT score::

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n[i])

where ``Q`` is the best value reachable from a node, ``P`` is a rank prior,
``n`` is the visit count, ``T`` the total visits and ``scale`` the reward spread.
Selected nodes whose lineages overlap are de-duplicated within a batch so a
single step explores diverse parents.

Unlike the reference (which used a Ray detached actor and per-step files keyed by
a Tinker log path), this archive lives inside slime's ``RolloutManager`` actor
and persists to ``<save>/ttt_archive/archive_step_XXXXXX.json``. The selection
math is preserved so behaviour matches the paper.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Sequence

import numpy as np

from .state import State


class DiscoveryArchive:
    def __init__(
        self,
        save_dir: str,
        *,
        batch_size: int,
        create_initial_states,
        max_buffer_size: int = 1000,
        puct_c: float = 1.0,
        topk_children: int = 2,
    ):
        self.save_dir = save_dir
        self.batch_size = batch_size
        self._create_initial_states = create_initial_states
        self.max_buffer_size = max_buffer_size
        self.puct_c = float(puct_c)
        self.topk_children = topk_children

        self._states: list[State] = []
        self._initial_ids: set[str] = set()
        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T: int = 0
        self._lock = threading.Lock()
        self._last_sampled: list[State] = []
        self._current_step = 0

        os.makedirs(self.save_dir, exist_ok=True)

    # ---- persistence ---------------------------------------------------
    def _path(self, step: int) -> str:
        return os.path.join(self.save_dir, f"archive_step_{step:06d}.json")

    def _save(self, step: int) -> None:
        store = {
            "step": step,
            "states": [s.to_dict() for s in self._states],
            "initial_ids": sorted(self._initial_ids),
            "puct_n": self._n,
            "puct_m": self._m,
            "puct_T": self._T,
        }
        tmp = self._path(step) + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(store, f)
        os.replace(tmp, self._path(step))

    def load_from_step(self, step: int) -> bool:
        path = self._path(step)
        if not os.path.exists(path):
            return False
        with open(path) as f:
            store = json.load(f)
        self._states = [State.from_dict(s) for s in store.get("states", [])]
        self._initial_ids = set(store.get("initial_ids", []))
        self._n = store.get("puct_n", {})
        self._m = store.get("puct_m", {})
        self._T = int(store.get("puct_T", 0))
        self._current_step = step
        return True

    def maybe_seed(self) -> None:
        """Populate the archive with initial states if it is empty."""
        if self._states:
            return
        seeds = self._create_initial_states(self.batch_size)
        for s in seeds:
            self._initial_ids.add(s.id)
            self._states.append(s)
        self._save(self._current_step)

    # ---- selection (PUCT) ---------------------------------------------
    def _construction_key(self, s: State):
        if s.construction is not None:
            try:
                return tuple(np.asarray(s.construction).reshape(-1).tolist())
            except Exception:
                return str(s.construction)
        return s.code or None

    def _lineage(self, s: State, children_map: dict[str, set[str]]) -> set[str]:
        lineage = {s.id} | {str(p["id"]) for p in (s.parents or []) if p.get("id")}
        queue, visited = [s.id], {s.id}
        while queue:
            sid = queue.pop(0)
            for cid in children_map.get(sid, []):
                if cid not in visited:
                    visited.add(cid)
                    lineage.add(cid)
                    queue.append(cid)
        return lineage

    def _children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for s in self._states:
            for p in (s.parents or []):
                pid = p.get("id")
                if pid:
                    children.setdefault(str(pid), set()).add(s.id)
        return children

    def sample(self, num_states: int) -> list[State]:
        with self._lock:
            self.maybe_seed()
            candidates = list(self._states)
            vals = np.array([float(s.value if s.value is not None else -np.inf) for s in candidates])
            non_initial = np.array([s.id not in self._initial_ids for s in candidates])
            v = vals[non_initial] if non_initial.any() else vals
            scale = float(max(np.max(v) - np.min(v), 1e-6)) if v.size else 1.0

            # rank-based prior
            ranks = np.argsort(np.argsort(-vals))
            prior = (len(vals) - ranks).astype(np.float64)
            prior = prior / prior.sum()
            sqrtT = np.sqrt(1.0 + self._T)

            scored = []
            for i, s in enumerate(candidates):
                n = self._n.get(s.id, 0)
                q = self._m.get(s.id, vals[i]) if n > 0 else vals[i]
                bonus = self.puct_c * scale * prior[i] * sqrtT / (1.0 + n)
                scored.append((q + bonus, vals[i], s))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

            picked: list[State] = []
            if num_states > 1:
                children_map = self._children_map()
                blocked: set[str] = set()
                for _, _, s in scored:
                    if s.id in blocked:
                        continue
                    picked.append(s)
                    blocked |= self._lineage(s, children_map)
                    if len(picked) >= num_states:
                        break
                # backfill if lineage dedup left us short
                if len(picked) < num_states:
                    for _, _, s in scored:
                        if s not in picked:
                            picked.append(s)
                        if len(picked) >= num_states:
                            break
            else:
                picked = [scored[0][2]] if scored else []

            # if still short (tiny archive), repeat best
            while len(picked) < num_states and picked:
                picked.append(picked[len(picked) % len(picked)])

            self._last_sampled = picked
            return picked

    # ---- update (expansion) -------------------------------------------
    def update(self, children: Sequence[State], parents: Sequence[State], step: int | None = None) -> int:
        """Add newly discovered valid children, keeping top-k per parent."""
        with self._lock:
            # PUCT bookkeeping over all attempts.
            parent_max: dict[str, float] = {}
            parent_obj: dict[str, State] = {}
            for child, parent in zip(children, parents):
                if child.value is None:
                    continue
                parent_max[parent.id] = max(parent_max.get(parent.id, -np.inf), float(child.value))
                parent_obj[parent.id] = parent
            for pid, y in parent_max.items():
                self._m[pid] = max(self._m.get(pid, y), y)
                anc = [pid] + [str(p["id"]) for p in (parent_obj[pid].parents or []) if p.get("id")]
                for aid in anc:
                    self._n[aid] = self._n.get(aid, 0) + 1
                self._T += 1

            # top-k children per parent
            by_parent: dict[str, list[tuple[State, State]]] = {}
            for child, parent in zip(children, parents):
                if child.value is None:
                    continue
                by_parent.setdefault(parent.id, []).append((child, parent))

            existing = {self._construction_key(s) for s in self._states}
            existing.discard(None)
            added = 0
            for pairs in by_parent.values():
                pairs.sort(key=lambda cp: cp[0].value if cp[0].value is not None else -np.inf, reverse=True)
                for child, parent in pairs[: self.topk_children] if self.topk_children > 0 else pairs:
                    key = self._construction_key(child)
                    if key is not None and key in existing:
                        continue
                    child.parent_values = ([parent.value] + parent.parent_values) if parent.value is not None else []
                    child.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents
                    self._states.append(child)
                    if key is not None:
                        existing.add(key)
                    added += 1

            self._prune()
            if step is not None:
                self._current_step = step
            self._save(self._current_step)
            return added

    def _prune(self) -> None:
        if len(self._states) <= self.max_buffer_size:
            return
        order = list(np.argsort([s.value if s.value is not None else -np.inf for s in self._states])[::-1])
        keep = {i for i, s in enumerate(self._states) if s.id in self._initial_ids}
        for i in order:
            if len(keep) >= self.max_buffer_size:
                break
            keep.add(i)
        self._states = [self._states[i] for i in sorted(keep)]

    # ---- logging helpers ----------------------------------------------
    def best_state(self) -> State | None:
        valid = [s for s in self._states if s.value is not None]
        return max(valid, key=lambda s: s.value) if valid else None

    def stats(self) -> dict:
        vals = np.array([s.value for s in self._states if s.value is not None], dtype=np.float64)
        out = {"ttt/archive_size": len(self._states), "ttt/visits_T": self._T}
        if vals.size:
            out.update(
                {
                    "ttt/archive_value_max": float(vals.max()),
                    "ttt/archive_value_mean": float(vals.mean()),
                    "ttt/archive_value_min": float(vals.min()),
                }
            )
        return out
