# slime-ttt

Portable, locally runnable reproduction of **TTT-Discover / Learning to
Discover at Test Time** on top of [slime](https://github.com/THUDM/slime),
Megatron, and SGLang.

The repository implements the core discovery algorithm and task environments
without requiring Tinker. It supports all five domains from the paper:

- mathematical discovery: Erdős minimum overlap, AC1, AC2, circle packing 26/32;
- GPU kernel engineering: TriMul and MLA Decode hardware adaptations;
- algorithm engineering: AHC039 and AHC058;
- biological discovery: single-cell RNA-seq denoising.

> The original paper trains LoRA adapters through Tinker. This port uses slime
> full-parameter training. The discovery loop, PUCT archive, adaptive-entropic
> objective, token-level KL shaping, importance ratios, and two-phase generation
> are aligned with the released implementation; the training backend differs.

## Reproduction status

The local campaign used Qwen3-4B on **8×A100 80GB**, plus 128 CPU cores and
755GB RAM. Headline results:

| Domain | Task | Local result | Status |
|---|---|---:|---|
| Math | Erdős minimum overlap | 0.381659 | 10 steps, 1,280 rollouts |
| Math | AC1 | 1.691701 | 5 steps, 640 rollouts |
| Math | AC2 | 0.796296 | 5 steps, 640 rollouts |
| Math | Circle packing 26 | 2.438966 | 5 steps, 640 rollouts |
| Math | Circle packing 32 | 2.666667 | 5 steps, 640 rollouts |
| GPU | TriMul A100 | 8/24 valid; seed 903.62 μs | end-to-end training |
| GPU | MLA Decode A100 | 4/4 valid; seed 539.462 μs | end-to-end smoke |
| AHC | AHC039 public replay | 145/150 AC | released artifact |
| AHC | AHC058 public replay | 150/150 AC | released artifact |
| Biology | Pancreas TTT | 4/4 valid, archive 1→3 | training update saved |
| Biology | PBMC held-out | 0.708497 | paper reports 0.71 |
| Biology | Tabula held-out | 0.734980 | paper reports 0.73 |

See [the complete reproduction table](docs/TTT_REPRODUCTION_RESULTS.md) for
budgets, evidence paths, official values, and exact comparability notes.

The A100 kernel measurements are hardware adaptations, not substitutes for the
paper's H100/H200/MI300X scores. Public AHC inputs are not the hidden AtCoder
evaluation set.

## How it works

At each test-time training step:

1. sample parent solutions from a persistent PUCT archive;
2. prompt the model with the problem and best-so-far solution;
3. generate a group of candidates with SGLang;
4. execute the official or ported task verifier;
5. update the archive and compute adaptive-entropic advantages;
6. apply the token-level KL-shaped policy update in Megatron;
7. synchronize weights and continue discovery.

Important alignment details are documented in
[TTT_PAPER_ALIGNMENT_AUDIT.md](docs/TTT_PAPER_ALIGNMENT_AUDIT.md).

## Quick start

Clone this repository and the official released reference as siblings:

```bash
git clone https://github.com/hqhq1025/slime-ttt.git
git clone https://github.com/test-time-training/discover.git ttt-discover-official
cd slime-ttt
```

Choose a large persistent storage directory. Models, datasets, checkpoints,
logs, and caches are deliberately excluded from Git:

```bash
export TTT_ROOT="$PWD"
export TTT_STORAGE_ROOT=/path/to/large/storage/ttt-storage
export TTT_OFFICIAL_ROOT="$(dirname "$PWD")/ttt-discover-official"

# Point these at the runtime installed on the target machine.
export TTT_VENV=/path/to/python-venv
export CUDA_HOME=/usr/local/cuda
export MEGATRON_PATH=/path/to/Megatron-LM
export SGLANG_SOURCE=/path/to/sglang/python

source local/env.sh
python -m pip install -e slime --no-deps
```

Prepare Qwen3-4B and run the smallest end-to-end job:

```bash
bash local/prepare_qwen3_4b.sh
bash local/run_ttt_erdos_smoke.sh
```

More launchers:

```bash
bash local/run_ttt_ac1_smoke.sh
bash local/run_ttt_domain_scale.sh ac2
bash local/run_ttt_domain_scale.sh circle26
bash local/run_ttt_trimul_a100_smoke.sh
bash local/run_ttt_mla_decode_a100_smoke.sh
bash local/run_ttt_denoising_smoke.sh
```

Read the [portable setup and execution runbook](docs/PORTABLE_TTT_RUNBOOK.md)
before moving to a new cluster. The
[Apptainer guide](docs/APPTAINER_GUIDE.md) is recommended when system CUDA and
compiled Python packages differ across machines.

## Repository layout

```text
slime/examples/ttt_discover/
├── archive.py                 # PUCT discovery archive
├── advantage.py               # official KL-shaped token advantages
├── ttt_rollout.py             # two-phase generation and evaluation loop
├── envs/                      # math, AHC, GPU, and biology tasks
├── trimul_eval.py             # local A100 TriMul evaluator
├── mla_decode_a100_eval.py    # local A100 MLA evaluator
└── tests/                     # alignment and evaluator contract tests

local/
├── env.sh                     # portable path/caching configuration
├── prepare_qwen3_4b.sh
├── run_ttt_*                  # smoke and scale launchers
└── evaluate_*                 # released/public artifact evaluators

docs/
├── PORTABLE_TTT_RUNBOOK.md
├── TTT_PAPER_ALIGNMENT_AUDIT.md
└── TTT_REPRODUCTION_RESULTS.md
```

## Validation

The published branch was checked with:

```bash
PYTHONPATH="$PWD/slime" "$TTT_VENV/bin/python" -m pytest -s -o addopts='' \
  slime/examples/ttt_discover/tests
```

The combined portable suite passes 32 environment/evaluator tests. Shell
launchers also pass `bash -n`, edited Python files pass `py_compile`, and the
machine-readable result tables are checked for consistency.

## Scaling beyond the local run

For a faithful gpt-oss-120B campaign matching the paper's 50×512-rollout,
32K-context setting, plan around **32×H100/H200 80GB for 48 hours**, 128–256
CPU cores, at least 512GB RAM, fast shared storage, and InfiniBand. An 8×A100
node is well suited to functional validation and Qwen3-4B experiments, but not
to reproducing the official 120B training budget.

Legacy Frontier-CS and multi-node Hyak launchers remain under `scripts/` for
reference. Frontier-CS requires its external problem assets and judge service;
it is not included in the verified paper-task table.

## Safety

Candidate programs are generated and executed during evaluation. Use isolated
compute nodes or containers, keep credentials outside the runtime, and do not
run untrusted candidates on machines containing sensitive data.

## References

- TTT-Discover official release: <https://github.com/test-time-training/discover>
- slime: <https://github.com/THUDM/slime>
- Reproduction evidence: [docs/TTT_REPRODUCTION_RESULTS.md](docs/TTT_REPRODUCTION_RESULTS.md)
