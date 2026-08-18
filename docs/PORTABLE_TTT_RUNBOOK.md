# Portable TTT-Discover runbook

This fork contains the slime implementation, local evaluators, launchers, and
the evidence table used for the A100 reproduction. Model weights, datasets,
official released artifacts, checkpoints, and dependency source trees are
intentionally not committed.

## 1. Clone code and the official reference

```bash
git clone https://github.com/hqhq1025/slime-ttt.git
git clone https://github.com/test-time-training/discover.git ttt-discover-official
cd slime-ttt
```

The default layout expects the two repositories to be siblings. Every path can
instead be overridden through the environment.

## 2. Choose persistent storage

```bash
export TTT_ROOT="$PWD"
export TTT_STORAGE_ROOT=/path/on/large/shared/storage/ttt-storage
export TTT_OFFICIAL_ROOT="$(dirname "$PWD")/ttt-discover-official"
export TTT_AHC_CACHE_DIR="$TTT_STORAGE_ROOT/ahc-cache/extracted/cache"
```

`TTT_STORAGE_ROOT` receives models, checkpoints, logs, Ray temporary files,
datasets, and caches. Do not place it on a small root filesystem.

## 3. Supply the slime runtime

Use a CUDA/PyTorch/slime environment compatible with the hardware. The native
A100 reproduction used Python 3.12, CUDA 12.8, PyTorch 2.9, Ray, SGLang,
Megatron-LM, Transformer Engine, FlashAttention, Triton, and the packages in
`slime/requirements.txt`.

Point the portable launcher at those components:

```bash
export TTT_VENV=/path/to/venv
export CUDA_HOME=/usr/local/cuda
export MEGATRON_PATH=/path/to/Megatron-LM
export SGLANG_SOURCE=/path/to/sglang/python
source local/env.sh
python -m pip install -e slime --no-deps
```

The upstream container/Apptainer route described in `docs/APPTAINER_GUIDE.md`
is preferable when moving between clusters with different system libraries.

## 4. Prepare Qwen3-4B and run smoke tests

```bash
bash local/prepare_qwen3_4b.sh
bash local/run_ttt_erdos_smoke.sh
bash local/run_ttt_ac1_smoke.sh
bash local/run_ttt_trimul_a100_smoke.sh
bash local/run_ttt_mla_decode_a100_smoke.sh
```

All launchers accept environment overrides. Important ones are `NUM_GPUS`,
`TTT_NUM_ROLLOUT`, `TTT_ROLLOUT_BATCH_SIZE`, `TTT_SAMPLES_PER_PROMPT`,
`TTT_RESPONSE_LEN`, and `TTT_MAX_TOKENS_PER_GPU`.

For the medium mathematics reproduction (five steps and 640 rollouts/task):

```bash
bash local/run_ttt_domain_scale.sh ac1
bash local/run_ttt_domain_scale.sh ac2
bash local/run_ttt_domain_scale.sh circle26
bash local/run_ttt_domain_scale.sh circle32
```

## 5. Optional domain-specific assets

- AHC039/058 require the released ALE-Bench cache at `TTT_AHC_CACHE_DIR`.
- Denoising uses a separate Python 3.11 scientific environment. Set
  `TTT_DENOISING_PYTHON`, `OPENPROBLEMS_PANCREAS_PATH`, and
  `OPENPROBLEMS_CACHE_DIR` before `local/run_ttt_denoising_smoke.sh`.
- TriMul and MLA A100 are hardware adaptations. Exact paper comparison requires
  H100/H200 and, for final MLA selection, MI300X.

## 6. Validate the checkout

```bash
PYTHONPATH="$PWD/slime" "$TTT_VENV/bin/python" -m pytest -s -o addopts='' \
  slime/examples/ttt_discover/tests
```

See `docs/TTT_REPRODUCTION_RESULTS.md` for measured results and comparability
notes. Public AHC replays and A100 kernel numbers are not substitutes for the
paper's hidden or target-hardware scores.
