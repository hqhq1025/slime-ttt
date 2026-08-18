# TTT-Discover local reproduction tracker

Snapshot: 2026-08-17. Official reference: `test-time-training/discover` commit
`6c40e82dab9d5de7416ac873ad5cd3106084aaed` and arXiv:2601.16175.

Status meanings:

- **completed**: end-to-end local TTT training and verifier completed.
- **env-ready**: official verifier contract is ported and unit-tested; training is not complete.
- **smoke-completed**: one end-to-end training update completed; not enough budget for a result comparison.
- **evaluator-completed**: the released/public evaluator and seed or baseline pass locally; training produced no valid child yet.
- **artifact-smoke**: released final artifact passes a local syntax/import check, but the official benchmark was not run.
- **blocked**: required evaluator, target hardware, or dataset is not locally ready.

## Results

| Domain | Task | Direction | Official result | Local result | Local budget | Status | Evidence / next requirement |
|---|---|---:|---:|---:|---|---|---|
| Mathematics | Erdős minimum overlap | ↓ | 0.380876 (gpt-oss-120B) | **0.381659** (Qwen3-4B) | 8×A100 80GB, 10 steps, 1,280 rollouts, ~51 min / 6.8 GPU-h | **completed** | Ray succeeded; archive 1→137. Best improved from scale step 0 0.382321 to 0.381659. |
| Mathematics | AC1 autocorrelation | ↓ | 1.50287 (gpt-oss-120B) | **1.691701** (Qwen3-4B) | 8×A100 80GB, 5 steps, 640 rollouts | **completed** | Ray succeeded; archive 1→81. Improved 1.953848→1.691701 (13.4%). Steps 3–4 resumed the archive from base weights after the verifier-stall interruption. |
| Mathematics | AC2 autocorrelation | ↑ | 0.9591 (gpt-oss-120B) | **0.796296** (Qwen3-4B) | 8×A100 80GB, 5 steps, 640 rollouts | **completed** | Ray succeeded; archive 1→72; 117/128 valid in the final step. Reached 83.0% of the official result. |
| Mathematics | Circle packing n=26 | ↑ | 2.635983 (Qwen3-8B) | **2.438966** (Qwen3-4B) | 8×A100 80GB, 5 steps, 640 rollouts | **completed** | Ray succeeded after archive empty-construction dedup fix; archive 8→58, final step 54/128 valid. Reached 92.5% of official. |
| Mathematics | Circle packing n=32 | ↑ | 2.939572 (Qwen3-8B) | **2.666667** (Qwen3-4B) | 8×A100 80GB, 5 steps, 640 rollouts | **completed** | Ray succeeded; archive 8→71, final step 70/128 valid. Reached 90.7% of official. |
| GPU kernels | TriMul A100 | ↓ μs | 2198.2 (gpt-oss-120B) | **903.62 μs seed; 1180.79 μs best generated child** | 8×A100, Qwen3-4B, local warm-start, 3 steps / 24 rollouts, ~19 min | **completed** | Ray succeeded; 8/24 valid, no truncation, archive 1→7; every step produced valid kernels, but none beat the seed. Released code needed the missing-mask default fix (11/18→18/18); full 7-case A100 seed benchmark was 2109.50 μs. A100 is not directly comparable to the paper's H100 reward. |
| GPU kernels | TriMul H100/B200/MI300X | ↓ μs | 1161.2 / 914.2 / 1555.7 | unavailable | no target GPU | **blocked** | Request H100/H200; B200/MI300X needed only for exact target reports. |
| GPU kernels | MLA Decode A100 smoke | ↓ μs | instance-dependent | **539.462 μs seed; 538.830 μs best generated child** | 8×A100, Qwen3-4B, local warm-start, 1 step / 4 rollouts | **completed** | Ray succeeded; 4/4 valid, no truncation, archive 1→3, grad norm 1.946624. The apparent 0.12% difference is within three-repeat timing noise and is not claimed as an optimization. The two official small correctness shapes pass; this A100 smoke is not comparable to the official H200/MI300X leaderboard. |
| GPU kernels | MLA Decode H200/MI300X | ↓ μs | instance-dependent | unavailable | no target GPU | **blocked** | Paper trained on H200 and selected on MI300X; request both classes or use the official remote evaluator for exact reproduction. |
| Algorithm engineering | AHC039 | ↑ | 567,062 | released artifact: **145/150 AC**, accepted-only mean **3,802.545**, aggregate **551,369**; strict aggregate **0** | Official released C++; public seeds 0–149; official tester; 302.47 s wall. Training smoke: Qwen3-4B, 1 step / 4 rollouts | **evaluator-completed** | Five first-pass runs (17, 18, 100, 104, 132) hit nondeterministic `SIGFPE` from modulo-zero when `rectPool` is empty; each passed 3/3 when rerun alone. Main result remains the unmodified first pass. Public replay is not comparable to the hidden aggregate. |
| Algorithm engineering | AHC058 | ↑ | 848,414,228 | released artifact: **150/150 AC**, mean **5,667,059.807**, aggregate **850,058,971** | Official released C++; public seeds 0–149; official tester; 64 workers; 10.09 s wall. Training smoke: Qwen3-4B, 1 step / 4 rollouts | **evaluator-completed** | Full released-artifact public replay succeeded. The public seed set differs from AtCoder's hidden evaluation, so the aggregate is not evidence of exceeding the paper result. Training smoke produced 0/4 valid generated programs. |
| Biology | scRNA-seq denoising | ↑ | PBMC 0.71; Tabula 0.73 | **PBMC 0.708497; Tabula 0.734980** (released artifact); pancreas TTT smoke completed | Qwen3-4B, 8×A100, 1 step / 4 rollouts; exact OpenProblems held-out protocol on CPU | **smoke-completed** | Pancreas end-to-end TTT: 4/4 valid, archive 1→3, grad norm 7.872218, checkpoint saved; best remained the MAGIC seed (MSE 0.231412). Both released-artifact headline results reproduce. |

## Erdős interpretation

The local 4B run is not a reproduction of the paper's final number: its bound is
0.000783 above the official 120B result (about 0.21% of the reported metric), and
also above the 0.380927 human bound. It *is* a successful reproduction of the
learning loop: the end-to-end job completed without OOM/NaN, produced 1,280
verified rollouts, expanded the archive to 137 states, and improved during the
10-step scale run.

## Biology interpretation

The released artifact reproduces both headline held-out results with the exact
OpenProblems notebook protocol. PBMC gives MSE 0.153902 and Poisson 0.047520;
relative to no-denoising/perfect baselines these normalize to 0.432041 and
0.984953, for a mean score of **0.708497** (reported 0.71). Tabula Muris lung
gives MSE 0.135783 and Poisson 0.029177, normalized to 0.482004 and 0.987955,
for a mean score of **0.734980** (reported 0.73). Both satisfy the 0.97
normalized-Poisson constraint. PBMC used 50.8 seconds wall / 3.69 GiB peak RSS;
Tabula used 13 minutes 32 seconds / 75.5 GiB peak RSS.

The separate pancreas/inDrop1 training evaluator also runs locally. There the
released artifact improves MSE from the MAGIC seed's 0.231412 to 0.165521
(28.47%) while improving normalized Poisson from 0.977048 to 0.995564; its
mean-normalized score is 0.726187. That pancreas number is a training-evaluator
result and should not be substituted for either held-out headline score.

The local Qwen3-4B pancreas TTT smoke also completed one full rollout, training,
weight-update, archive, and checkpoint cycle on 8×A100. All 4/4 generated
candidates were valid and the archive grew from 1 to 3, with gradient norm
7.872218. The generated candidates did not beat the MAGIC seed, whose MSE
remained 0.2314119079, so this validates the training path rather than claiming
an optimization result.

## Resource request backed by this table

For a faithful gpt-oss-120B run (50 × 512 rollouts, 32K context, LoRA rank 32),
request **32×H100/H200 80GB for 48 hours**, plus 128–256 CPU cores, at least
512GB RAM, fast shared storage and InfiniBand. A 16-GPU allocation is useful for
a reduced-budget run; 8 GPUs are primarily for functional and small-model runs.

## Evidence paths

- Erdős best: `checkpoints/qwen3-4b-ttt-erdos-8x16x10/ttt_best/best.json`
- Erdős candidates: `checkpoints/qwen3-4b-ttt-erdos-8x16x10/ttt_rollouts/`
- Erdős archive: `checkpoints/qwen3-4b-ttt-erdos-8x16x10/ttt_archive/`
- AC environments: `slime/examples/ttt_discover/envs/ac_inequalities.py`
- AC1 smoke: `checkpoints/qwen3-4b-ttt-ac1-smoke/`
- AC1 scale: `checkpoints/qwen3-4b-ttt-ac1-8x16x5/`
- AC2 scale: `checkpoints/qwen3-4b-ttt-ac2-8x16x5/`
- Circle environments: `slime/examples/ttt_discover/envs/circle_packing.py`
- Circle 26 scale: `checkpoints/qwen3-4b-ttt-circle26-8x16x5/`
- Circle 32 scale: `checkpoints/qwen3-4b-ttt-circle32-8x16x5/`
- TriMul environment/evaluator: `slime/examples/ttt_discover/envs/trimul_a100.py` and `slime/examples/ttt_discover/trimul_eval.py`
- TriMul smoke: `checkpoints/qwen3-4b-ttt-trimul-a100-smoke/`
- TriMul 3-step run: `checkpoints/qwen3-4b-ttt-trimul-a100-8x8x3/`
- MLA Decode A100 environment/evaluator: `slime/examples/ttt_discover/envs/mla_decode_a100.py` and `slime/examples/ttt_discover/mla_decode_a100_eval.py`
- MLA Decode A100 smoke: `checkpoints/qwen3-4b-ttt-mla-decode-a100-smoke/`
- MLA Decode A100 best child: `checkpoints/qwen3-4b-ttt-mla-decode-a100-smoke/ttt_best/best.json`
- AHC public evaluator: `slime/examples/ttt_discover/ahc_judge.py`
- AHC039 smoke: `checkpoints/qwen3-4b-ttt-ahc039-smoke/`
- AHC039 released public replay: `local/evaluate_ahc039_released_public.py`, `local/run_ahc039_released_public.sh`, `docs/ahc039_released_public_150.json`, and `.csv`
- AHC039 failure diagnosis and independent retries: `docs/AHC039_RELEASED_PUBLIC_REPLAY.md` and `docs/ahc039_released_public_retry_{1,2,3}.{json,csv}`
- AHC058 smoke: `checkpoints/qwen3-4b-ttt-ahc058-smoke/`
- AHC058 released public replay: `local/evaluate_ahc058_released_public.py`, `local/run_ahc058_released_public.sh`, `docs/ahc058_released_public_150.json`, and `.csv`
- Denoising evaluator launcher: `local/eval_ttt_denoising.sh`
- Denoising evaluator: `local/evaluate_denoising.py`
- Denoising Qwen3-4B pancreas TTT smoke: `checkpoints/qwen3-4b-ttt-denoising-pancreas-smoke-fixed/`
- Denoising pancreas smoke best/archive/rollouts: `checkpoints/qwen3-4b-ttt-denoising-pancreas-smoke-fixed/ttt_best/`, `ttt_archive/`, and `ttt_rollouts/`
- Denoising held-out evaluator: `local/evaluate_denoising_heldout.py`
- Denoising PBMC held-out result: `results/denoising-heldout/pbmc-released.json`
- Denoising Tabula held-out result: `results/denoising-heldout/tabula-released.json`
- Denoising seed log: `/data/haoqing/denoising-results/pancreas_seed.log`
- Denoising released-artifact log: `/data/haoqing/denoising-results/pancreas_released_ttt_json.log`
- Machine-readable snapshot: `docs/ttt_reproduction_results.csv` and `.json`
