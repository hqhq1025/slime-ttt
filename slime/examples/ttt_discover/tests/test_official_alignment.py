"""Regression tests for semantics copied from the official TTT-Discover code."""

from __future__ import annotations

import tempfile

import torch

from examples.ttt_discover.advantage import shape_advantages
from examples.ttt_discover.archive import DiscoveryArchive
from examples.ttt_discover.environment import extract_last_code_block
from examples.ttt_discover.state import State


def _seeds(n: int) -> list[State]:
    return [State(value=float(i), construction=[float(i)]) for i in range(n)]


def test_truncated_code_block_is_accepted():
    text = "reasoning\n```python\ndef run():\n    return 1"
    assert extract_last_code_block(text) == "def run():\n    return 1"


def test_last_complete_code_block_wins():
    text = "```python\nold = 1\n```\ntext\n```python\nnew = 2\n```"
    assert extract_last_code_block(text) == "new = 2"


def test_failed_expansion_updates_puct_visits():
    with tempfile.TemporaryDirectory() as save_dir:
        archive = DiscoveryArchive(save_dir, batch_size=1, create_initial_states=_seeds)
        parent = archive.sample(1)[0]
        archive.update([], [], step=0, attempted_parents=[parent])
        assert archive._T == 1
        assert archive._n[parent.id] == 1
        assert archive._m.get(parent.id) is None


def test_empty_construction_uses_code_for_archive_deduplication():
    with tempfile.TemporaryDirectory() as save_dir:
        archive = DiscoveryArchive(save_dir, batch_size=1, create_initial_states=_seeds)
        first = State(value=2.1, construction=[], code="first packing")
        second = State(value=2.2, construction=[], code="second packing")
        assert archive._construction_key(first) == "first packing"
        assert archive._construction_key(second) == "second packing"


def test_mapping_construction_has_a_stable_hashable_archive_key():
    with tempfile.TemporaryDirectory() as save_dir:
        archive = DiscoveryArchive(save_dir, batch_size=1, create_initial_states=_seeds)
        first = State(value=-0.23, construction={"mse": 0.23, "poisson": 0.04})
        second = State(value=-0.22, construction={"poisson": 0.04, "mse": 0.23})
        first_key = archive._construction_key(first)
        assert hash(first_key) is not None
        assert first_key == archive._construction_key(second)


def test_nested_container_construction_has_a_canonical_archive_key():
    with tempfile.TemporaryDirectory() as save_dir:
        archive = DiscoveryArchive(save_dir, batch_size=1, create_initial_states=_seeds)
        first = State(construction={"nested": [1, (2, {4, 3})], "flag": True})
        second = State(construction={"flag": True, "nested": [1, (2, {3, 4})]})
        assert archive._construction_key(first) == archive._construction_key(second)
        assert hash(archive._construction_key(first)) is not None


def test_scalar_construction_key_behavior_is_unchanged():
    with tempfile.TemporaryDirectory() as save_dir:
        archive = DiscoveryArchive(save_dir, batch_size=1, create_initial_states=_seeds)
        assert archive._construction_key(State(construction="value")) == ("value",)
        assert archive._construction_key(State(construction=3)) == (3,)


def test_kl_shaping_matches_official_formula():
    rewards = [2.0, -1.0]
    rollout = [torch.tensor([-1.0, -2.0]), torch.tensor([-0.5, -1.5])]
    ref = [torch.tensor([-1.5, -1.5]), torch.tensor([-1.0, -2.0])]
    masks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])]
    coef = 0.1

    got = shape_advantages(rewards, rollout, ref, masks, coef)
    diffs = [(rollout[i] - ref[i]) * masks[i] for i in range(2)]
    avg = sum(x.sum() for x in diffs) / sum(x.sum() for x in masks)
    expected = [
        torch.full_like(diffs[i], rewards[i]) + coef * masks[i] * (avg - diffs[i])
        for i in range(2)
    ]
    for actual, target in zip(got, expected, strict=True):
        torch.testing.assert_close(actual, target)
