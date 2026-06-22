"""Discovery state: one candidate solution and its lineage.

A ``State`` is the unit the discovery archive stores and samples from. It carries
the last program (``code``), the resulting object (``construction``, e.g. a numpy
array of values), the achieved ``value`` (higher = better, internally), the
captured ``stdout`` (``observation``), and a pointer chain to its parents so the
archive can run a tree search over edits.

``to_prompt`` is a faithful port of TTT-Discover's ``State.to_prompt``: it turns
the best-so-far solution into the in-context seed that conditions the next round
of generation. This is the "in-context" half of TTT — the weight-space half is
the RL update slime performs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def to_json_serializable(obj: Any) -> Any:
    """Best-effort conversion of numpy types to JSON-safe Python types."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj


@dataclass
class State:
    timestep: int = -1
    code: str = ""
    construction: Any = None
    value: float | None = None  # higher = better (envs that minimize store -score)
    observation: str = ""  # captured stdout of the program that produced this state
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestep": self.timestep,
            "code": self.code,
            "construction": to_json_serializable(self.construction),
            "value": self.value,
            "observation": self.observation,
            "parent_values": self.parent_values,
            "parents": self.parents,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            id=d.get("id") or str(uuid.uuid4()),
            timestep=d.get("timestep", -1),
            code=d.get("code", ""),
            construction=d.get("construction"),
            value=d.get("value"),
            observation=d.get("observation", ""),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
        )

    # ---- prompt conditioning (port of TTT-Discover State.to_prompt) ----
    def to_prompt(
        self,
        target: float | None,
        metric_name: str = "value",
        maximize: bool = True,
        language: str = "",
    ) -> str:
        ctx = f"You are iteratively optimizing {metric_name}."
        direction = "higher" if maximize else "lower"

        if self.code and self.code.strip():
            ctx += "\nHere is the last code we ran:\n"
            ctx += f"```{language}\n{self.code}\n```" if language else self.code
        else:
            ctx += "\nNo previous code available."

        # When there is no fixed cap (target is None, e.g. an unbounded objective Value),
        # frame it as "push as far as possible" instead of showing a misleading gap.
        no_cap = (
            f"\nThere is no fixed cap — make {metric_name} as {'large' if maximize else 'small'} "
            "as possible; further improvements are always generously rewarded."
        )
        if self.parent_values and self.value is not None and self.construction is not None:
            before = self.parent_values[0] if maximize else -self.parent_values[0]
            after = self.value if maximize else -self.value
            ctx += (
                f"\nHere is the {metric_name} before and after running the code above "
                f"({direction} is better): {before:.6f} -> {after:.6f}"
            )
            if target is not None:
                gap = target - after if maximize else after - target
                ctx += f"\nTarget: {target}. Current gap: {gap:.6f}. Further improvements will also be generously rewarded."
            else:
                ctx += no_cap
        elif self.value is not None:
            after = self.value if maximize else -self.value
            ctx += f"\nCurrent {metric_name}: {after:.6f}"
            if target is not None:
                gap = target - after if maximize else after - target
                ctx += f"\nTarget: {target}. Current gap: {gap:.6f}. Further improvements will also be generously rewarded."
            else:
                ctx += no_cap
        else:
            ctx += (
                f"\nTarget {metric_name}: {target}" if target is not None
                else f"\nOptimize for the {direction}est {metric_name} you can achieve."
            )

        if self.observation and self.observation.strip():
            stdout = self.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

        return ctx
