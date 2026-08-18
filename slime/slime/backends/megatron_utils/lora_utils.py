"""Lightweight access to Megatron-Bridge PEFT without importing its model zoo."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path


def get_bridge_lora_class():
    """Import Bridge's LoRA implementation while bypassing optional model-zoo deps.

    ``megatron.bridge.__init__`` eagerly registers every supported HF model and
    therefore requires ModelOpt, Transformers 5, and vision/audio packages that
    are irrelevant to dense Qwen LoRA.  PEFT itself has a narrow dependency
    surface, so expose the package path directly and import only that submodule.
    """
    try:
        lora_class = importlib.import_module("megatron.bridge.peft.lora").LoRA
        _patch_adapter_base_checkpoint_loading()
        return lora_class
    except ModuleNotFoundError as exc:
        if exc.name not in {"modelopt", "megatron.bridge"}:
            raise

    bridge_src = os.environ.get("MEGATRON_BRIDGE_PATH")
    if bridge_src:
        bridge_dir = Path(bridge_src) / "megatron/bridge"
    else:
        bridge_dir = Path(__file__).resolve().parents[4] / "Megatron-Bridge/src/megatron/bridge"
    if not (bridge_dir / "peft/lora.py").is_file():
        raise FileNotFoundError(f"Megatron-Bridge PEFT source not found under {bridge_dir}")

    # Remove a partially imported eager package after the ModelOpt failure.
    for name in list(sys.modules):
        if name == "megatron.bridge" or name.startswith("megatron.bridge."):
            del sys.modules[name]

    import megatron

    bridge = types.ModuleType("megatron.bridge")
    bridge.__path__ = [str(bridge_dir)]
    bridge.__package__ = "megatron.bridge"
    sys.modules["megatron.bridge"] = bridge
    setattr(megatron, "bridge", bridge)
    lora_class = importlib.import_module("megatron.bridge.peft.lora").LoRA
    _patch_adapter_base_checkpoint_loading()
    return lora_class


def _patch_adapter_base_checkpoint_loading() -> None:
    """Let an adapter-wrapped model read an older base-only dist checkpoint."""
    wrapper_module = importlib.import_module("megatron.bridge.peft.adapter_wrapper")
    wrapper_class = wrapper_module.AdapterWrapper
    if getattr(wrapper_class, "_slime_base_checkpoint_patch", False):
        return
    original = wrapper_class.sharded_state_dict

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        if os.environ.get("SLIME_TTT_LORA_BASE_CHECKPOINT_LOAD") == "1":
            return self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata)
        return original(self, prefix, sharded_offsets, metadata)

    wrapper_class.sharded_state_dict = sharded_state_dict
    wrapper_class._slime_base_checkpoint_patch = True
