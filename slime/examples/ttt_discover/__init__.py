"""Full-parameter TTT-Discover on slime.

Public surface (kept dependency-light so loading an env class or running the CPU
tests does not pull in SGLang/torch):
- ``TTTEnvironment``          : base class for new problems
- ``RewardResult``, ``State`` : data carried through the loop
- ``DiscoveryArchive``        : the PUCT tree-search archive
- ``entropic_reward_post_process`` : drop-in ``--custom-reward-post-process-path``
  for running on an *unmodified* slime (no core edits required)

The rollout entry point lives at ``examples.ttt_discover.ttt_rollout.generate_rollout``
(it imports SGLang); reference it by that full path in ``--rollout-function-path``.
"""

from .archive import DiscoveryArchive
from .environment import RewardResult, TTTEnvironment, extract_last_code_block
from .reward_post_process import entropic_reward_post_process
from .state import State

__all__ = [
    "TTTEnvironment",
    "RewardResult",
    "State",
    "DiscoveryArchive",
    "extract_last_code_block",
    "entropic_reward_post_process",
]
