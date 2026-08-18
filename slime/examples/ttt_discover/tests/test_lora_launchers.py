from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_lora_launchers_do_not_force_disable_cuda_graphs():
    launchers = [
        "local/run_ttt_erdos_lora_smoke.sh",
        "local/run_ttt_erdos_lora_scale.sh",
        "local/run_ttt_domain_lora_scale.sh",
        "local/run_ttt_trimul_lora_scale.sh",
        "local/run_ttt_cross_domain_lora_suite.sh",
    ]

    for relative_path in launchers:
        text = (REPO_ROOT / relative_path).read_text()
        assert "TTT_SGLANG_DISABLE_CUDA_GRAPH=1" not in text
        assert 'TTT_SGLANG_DISABLE_CUDA_GRAPH="${TTT_SGLANG_DISABLE_CUDA_GRAPH:-1}"' not in text


def test_base_launcher_keeps_cuda_graph_disable_as_explicit_opt_in():
    text = (REPO_ROOT / "local/run_ttt_erdos_smoke.sh").read_text()
    assert '${TTT_SGLANG_DISABLE_CUDA_GRAPH:-0}' in text
    assert "--sglang-disable-cuda-graph" in text


def test_base_launcher_captures_the_scaled_per_engine_decode_batch():
    text = (REPO_ROOT / "local/run_ttt_erdos_smoke.sh").read_text()
    assert 'SGLANG_CUDA_GRAPH_MAX_BS="${TTT_SGLANG_CUDA_GRAPH_MAX_BS:-64}"' in text
    assert '--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"' in text
