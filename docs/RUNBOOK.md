# RUNBOOK — full-parameter TTT-Discover (gpt-oss-20B, Frontier-CS) on Hyak

Everything lives under `/gscratch/zlab/lky04/slime-ttt`. Use **sbatch** (interactive jobs
are discouraged). Pitfalls behind each setting: see `DEBUG_LOG.md`.

## 0. Prereqs (already done, one-time)
- Container: `apptainer/images/slime.sif` (slime + Megatron + SGLang + TE; no pip needed).
- Model (converted): `models/gpt-oss-20b` (HF), `models/gpt-oss-20b-bf16` (SGLang + bridge),
  `models/gpt-oss-20b_torch_dist` (Megatron init / `--ref-load`).
  Re-create with `scripts/prepare_gptoss.sh` if ever needed.
- Problem data: `frontiercs/problems/<id>/` (statement, testdata, chk.cc) + `frontiercs/judge`.
- Modified slime + TTT plugin: `slime/` (custom rollout `examples/ttt_discover/ttt_rollout.py`,
  env `examples/ttt_discover/envs/frontiercs.py`, judge `frontiercs_judge.py`,
  entropic estimator in slime core).
- wandb key: `scripts/wandb_secret.sh` (chmod 600).

## 1. Launch (sbatch)
```bash
cd /gscratch/zlab/lky04/slime-ttt
sbatch scripts/submit.sbatch
# overrides (env passed via --export):
PROBLEM_ID=3 RBS=4 NSAMP=16 sbatch --export=ALL scripts/submit.sbatch
```
The job: requests 4xH200 nodes -> brings up a containerized multi-node Ray cluster
(`ray_up.sh`, SLURM-aware) -> runs training (`submit_train.sh` -> `_train_job.sh`) ->
cleans up Ray on exit (trap).

## 2. What runs (the TTT loop)
Per step: sample parent solutions from the discovery archive -> build prompts conditioned
on best-so-far -> generate `RBS x NSAMP` C++ candidates via SGLang -> score via the remote
judge (yanagiorigami.uk) -> entropic advantage -> **full-parameter** Megatron update ->
sync weights to SGLang -> repeat. Verified: best_raw rises (45.7 -> 76.6+), archive grows.

## 3. Monitor
- wandb: project `ttt-frontiercs-gptoss20b` (watch `ttt/best_raw_score`, `train/loss`,
  `ttt/frac_correct`, `ttt/archive_size`).
- Log: `tail -f logs/sbatch_<jobid>.out`  (or `ray job logs` if attaching).
- Best solution so far: `ckpts/ttt_best/best.json`; archive: `ckpts/ttt_archive/`.

## 4. Resume
Checkpoints every 5 steps in `ckpts/iter_XXXXXXX` (+ `latest_checkpointed_iteration.txt`).
Re-submitting auto-resumes: `_train_job.sh` adds `--load ckpts` only when that marker exists.

## 5. Stop
`scancel <jobid>` — the sbatch EXIT trap runs `ray_down.sh` to kill Ray/SGLang on all nodes.

## 6. Final working config (do not change without re-checking DEBUG_LOG.md §E)
| knob | value | why |
|---|---|---|
| nodes x gpus | 4 x 4 (16 H200) | the allocation |
| `--num-gpus-per-node` | **4** | default 8 != physical -> SGLang cross-node timeout |
| `--rollout-num-gpus-per-engine` | **1** | one gpt-oss engine per card |
| TP / PP / CP / EP | **4 / 1 / 1 / 4** | TP=4 fixes 26k-seq OOM; CP forbidden (sinks) |
| `--qkv-format` | **bshd** | gpt-oss learnable softmax forbids thd |
| `--use-dynamic-batch-size` | **off** (use `--micro-batch-size 1`) | required by bshd |
| `--attention-backend` | **fused** (cuDNN) | flash unsupported for sinks |
| NCCL/GLOO_SOCKET_IFNAME | **ens11f0np0** | cross-node fabric NIC |
| ray `--memory/--object-store-memory` | 460G / 40G per node | respect 512G/node grant |
| rollout-batch-size x n-samples | 4 x 16 (=64) | the "4x16"; lots of GPU headroom to scale |
| rollout-max-response-len | 26000 | token budget |
| advantage-estimator | entropic_adaptive_beta | the TTT discovery objective |

## 7. Knobs to tune later
- Scale group: `RBS`/`NSAMP` (GPU mem ~comfortable with TP=4).
- bshd pads to batch-max seq (compute waste) -> future: length bucketing / cap max-response-len.
- `--num-rollout` (default 50), `--save-interval` (5), `--lr` (1e-6).
