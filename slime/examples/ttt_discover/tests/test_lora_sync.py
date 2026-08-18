from types import SimpleNamespace

import torch

from slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
    _build_local_param_info_dict,
    _merge_lora_source_weight,
)


def _param(value):
    param = torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32))
    param.tensor_model_parallel = False
    param.partition_dim = -1
    param.partition_stride = 1
    return param


def test_lora_sync_metadata_flattens_wrappers_and_skips_adapters():
    prefix = "module.module.decoder.layers.0.self_attention.linear_qkv"
    raw = {
        f"{prefix}.to_wrap.weight": _param([[1.0, 2.0], [3.0, 4.0]]),
        f"{prefix}.adapter.linear_in.weight": _param([[1.0, 0.0], [0.0, 1.0]]),
        f"{prefix}.adapter.linear_out.weight": _param([[0.0, 0.0], [0.0, 0.0]]),
    }
    args = SimpleNamespace(ttt_lora_rank=2, ttt_lora_alpha=2)
    infos = _build_local_param_info_dict(args, raw, rank=0)
    assert list(infos) == [f"{prefix}.weight"]
    info = infos[f"{prefix}.weight"]
    assert info.attrs["source_name"] == f"{prefix}.to_wrap.weight"
    assert info.attrs["lora_A_name"].endswith("adapter.linear_in.weight")
    assert info.attrs["lora_B_name"].endswith("adapter.linear_out.weight")


def test_lora_sync_merges_rank_update_for_tp1(monkeypatch):
    prefix = "module.module.decoder.layers.0.mlp.linear_fc2"
    base_name = f"{prefix}.to_wrap.weight"
    a_name = f"{prefix}.adapter.linear_in.weight"
    b_name = f"{prefix}.adapter.linear_out.weight"
    base = _param([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    weights = {
        base_name: base,
        a_name: _param([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]),
        b_name: _param([[2.0, 0.0], [0.0, 4.0]]),
    }
    info = _build_local_param_info_dict(
        SimpleNamespace(ttt_lora_rank=2, ttt_lora_alpha=2), weights, rank=0
    )[f"{prefix}.weight"]
    monkeypatch.setattr(
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct.mpu.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct.mpu.get_tensor_model_parallel_group",
        lambda: None,
    )
    merged = _merge_lora_source_weight(info, base.detach(), weights)
    expected = base.detach() + weights[b_name].detach() @ weights[a_name].detach()
    torch.testing.assert_close(merged, expected)
