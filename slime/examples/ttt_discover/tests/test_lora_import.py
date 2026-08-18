from slime.backends.megatron_utils.lora_utils import get_bridge_lora_class


def test_lightweight_bridge_lora_import_avoids_optional_model_zoo():
    lora_class = get_bridge_lora_class()
    config = lora_class(dim=32, alpha=32)
    assert config.dim == config.alpha == 32
    assert config.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
