# slime-ttt on Hyak — full-parameter TTT-Discover for gpt-oss-20B / Frontier-CS

Everything for this project lives under **`/gscratch/zlab/lky04/slime-ttt`** (the
container, caches, model, problem data, checkpoints, logs). Nothing is written to
`$HOME` or outside this folder. The container is run with `--no-home` and all
caches (HF/XDG/triton/inductor/torch) are redirected here.

This deploys the TTT-Discover algorithm ("Learning to Discover at Test Time",
arXiv:2601.16175) on **slime** with **full-parameter** Megatron training (not the
paper's LoRA/Tinker) — the entropic discovery objective + a per-problem
discovery loop — applied to **Frontier-CS** competitive-programming problems with
**gpt-oss-20B**, on one node of 4×H200 (g3125).

## Directory layout
```
slime-ttt/
├── apptainer/images/slime.sif     # container (copied from socialrl; torch2.9/sglang/megatron)
├── slime/                         # modified slime v0.3.0 + examples/ttt_discover plugin
│   └── examples/ttt_discover/     # entropic estimator (core) + discovery loop + FrontierCS env
├── frontiercs/                    # Frontier-CS problems + judge/testlib (copied here)
│   ├── problems/<id>/             # statement.txt, config.yaml, testdata/, chk.cc
│   └── judge/, testlib.h
├── models/
│   ├── gpt-oss-20b/               # HF download (MXFP4)
│   ├── gpt-oss-20b-bf16/          # dequantized bf16 (for SGLang + conversion)  [prep]
│   └── gpt-oss-20b_torch_dist/    # Megatron torch_dist (train init / ref)       [prep]
├── cache/, hf_cache/              # all caches redirected here (never $HOME)
├── ckpts/                         # slime --load/--save
├── logs/                          # prepare + training logs
├── scripts/
│   ├── prepare_gptoss.sh          # download + bf16 + torch_dist (run on g3125)
│   ├── run-ttt-frontiercs-gptoss20b.sh   # << launch training (outer)
│   └── train_inner.sh            # the in-container command (edit hyperparams here)
└── README.md
```

## How to run (after model prep finishes)
On the compute node (g3125):
```bash
cd /gscratch/zlab/lky04/slime-ttt
# local judge (smoke, compile+run public test):
PROBLEM_ID=0 JUDGE_BACKEND=local bash scripts/run-ttt-frontiercs-gptoss20b.sh
# remote judge (real scoring) — once you give me the endpoint:
PROBLEM_ID=0 JUDGE_BACKEND=remote JUDGE_URL=http://<host>/score bash scripts/run-ttt-frontiercs-gptoss20b.sh
```
Tune training in `scripts/train_inner.sh` (group sizes, steps, parallelism, LR).

## What the loop does
Per step: sample parent solution(s) from a discovery archive → build a prompt
conditioned on the best-so-far C++ solution → generate a group of candidates via
SGLang → score each via the judge → entropic-advantage update (full-parameter) →
sync weights → repeat. Reward = judge points (normalized to [0,1]); the entropic
estimator concentrates the gradient on the best-in-group candidates.

## >>> JUDGE: the one thing left to finalize <<<
The reward backend is `examples/ttt_discover/frontiercs_judge.py`:
- `local`  : compiles + runs the **public** testdata + testlib checker (smoke only,
  binary-ish). Works now.
- `remote` : POSTs the candidate to your judge server. **Current assumed contract**
  (please confirm / correct):
      POST {JUDGE_URL}
      request : {"problem_id": str, "language": "cpp", "code": str}
      response: {"score": float, "max_score": float, "status": str}
  Once you describe the real protocol/URL I will update `remote_judge()` and the
  `--ttt-judge-*` args, then switch the run to `JUDGE_BACKEND=remote`.

## Open questions for you
1. Remote judge: URL, request/response schema, auth/token? (see above)
2. Which Frontier-CS problem id(s)? (TTT is single-problem; default PROBLEM_ID=0)
3. Reward shaping ok? (reward = judge_score / max_score in [0,1])
4. Scale: currently rollout-batch-size 4 × n-samples 8 = 32 rollouts/step, 50 steps.
   Paper uses up to 64/group; we can scale once the first run is stable.

## Setup status (verified)
- [x] Container reused (slime.sif) — my slime v0.3.0 + TTT plugin import cleanly inside it
- [x] Entropic-advantage unit tests pass inside the container
- [x] FrontierCS env smoke-tested (compile/run/checker path works)
- [x] Frontier-CS problem data copied into zlab
- [~] gpt-oss-20B prep running (download done; bf16 + torch_dist converting) — see logs/prepare_gptoss.log
- [ ] First full-parameter training launch (pending checkpoint; parallelism/memory may need tuning)
- [ ] Remote judge wired (pending your protocol)

## Notes
- All compute runs on g3125 inside slime.sif. Caches are bound into zlab; `--no-home`.
- gpt-oss MoE on 4×H200: expert-parallel=4. If you hit OOM, lower
  --max-tokens-per-gpu or n-samples-per-prompt, or add pipeline parallel in train_inner.sh.

## Update (setup complete)
- [x] gpt-oss-20B fully prepared: models/gpt-oss-20b (HF), gpt-oss-20b-bf16 (8 shards),
      gpt-oss-20b_torch_dist (Megatron dist ckpt: release/ + 8 .distcp + common.pt).
- [x] Full training command arg-parse validated inside the container (entry point
      train_ttt.py + all --ttt-* args + --advantage-estimator entropic_adaptive_beta).
- READY TO LAUNCH. Remaining before a real run: (1) your remote-judge protocol so
  reward is meaningful, (2) decide PROBLEM_ID + scale. A local-judge smoke run can be
  started any time with: PROBLEM_ID=0 JUDGE_BACKEND=local bash scripts/run-ttt-frontiercs-gptoss20b.sh
  (first GPU launch may need parallelism/mem tuning in scripts/train_inner.sh — EP=4,
  --max-tokens-per-gpu, --sglang-mem-fraction-static).

## Update 2 (judge WIRED + verified)
Remote judge protocol discovered & implemented in examples/ttt_discover/frontiercs_judge.py:
  POST https://yanagiorigami.uk/submit  {"pid": str, "lang": "cpp", "code": str}  -> {"sid": int}
  GET  https://yanagiorigami.uk/result/{sid}  -> {"status":"done","passed":bool,
       "result":str,"score":float,"scoreUnbounded":float,"cases":[...]}
  (must send a browser User-Agent; Cloudflare 403s python-urllib default.)
Verified from inside the container: a trivial solution to problem 0 -> "Wrong Answer", score 0, correctness 1.
Run scripts now default to JUDGE_BACKEND=remote, JUDGE_URL=https://yanagiorigami.uk.
Reward = judge "score"/100 (switch to raw with --ttt-judge-score-key scoreUnbounded for optimization problems).

## WORKING CONFIG (4x16 full loop verified 2026-06-15)
gpt-oss-20B, Frontier-CS, 16xH200 (4 nodes). Full TTT loop runs: rollout(64x26k + remote judge) -> entropic adv -> full-param Megatron update -> weight sync -> next rollout. best_raw improved 27.8 -> 46.3 over 1 step.
Critical multi-node/gpt-oss fixes (in scripts/_train_job.sh + ray_env.sh):
- --num-gpus-per-node 4   (default 8; must match physical GPUs/node or SGLang engines span nodes -> TCPStore timeout)
- NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=ens11f0np0  (fabric NIC for cross-node)
- --rollout-num-gpus-per-engine 1  (one gpt-oss engine per card)
- ray start --memory/--object-store-memory caps (respect 512G/node SLURM grant)
- --load only if a real checkpoint exists (else init from --ref-load)
- --attention-backend fused + --qkv-format bshd + NO --use-dynamic-batch-size
  (gpt-oss learnable softmax/sinks: flash unsupported; cuDNN fused needs bshd not thd)
Note: bshd pads to batch-max seq (compute waste on short samples); fine given big GPU headroom (~48/140GB used).

## FINAL STABLE CONFIG (verified continuous, 2026-06-15)
4x16 (rollout-batch-size 4 x n-samples 16), 26k token limit, gpt-oss-20B, 16xH200.
Ran 10+ steps continuously, best_raw 45.7 -> 76.6 (rising), archive 12 -> 122. Checkpoints every 5 steps.
Parallelism: TP=4, PP=1, CP=1, EP=4, ETP=1 (world=16, DP=4).
gpt-oss attention constraints (TE 2.10/cuDNN 9.16) due to learnable softmax (attention sinks):
  - learnable + thd            -> NO backend  => must use --qkv-format bshd (no --use-dynamic-batch-size)
  - learnable + context-parallel -> NO backend => CANNOT use CP
  - learnable + bshd + no CP   -> cuDNN FusedAttention works  (use --attention-backend fused)
  - expandable_segments allocator BREAKS SGLang TorchMemorySaver => do NOT set it
  - long-seq (26k) OOM fix = TP=4 (shard per-layer attn/logits), NOT CP, NOT expandable_segments
GPU mem ~comfortable with TP=4. bshd pads to batch-max seq (compute waste; optional future opt: length bucketing).
