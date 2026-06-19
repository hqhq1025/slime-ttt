# TTT-Discover on slime / gpt-oss-20B @ Hyak — Debug Log

Every pitfall hit bringing up **full-parameter TTT-Discover (gpt-oss-20B, Frontier-CS,
4 nodes x 4 H200) on slime**, with symptom -> root cause -> fix. Ordered by the stage
it bit us. Final working config + run flow: see `RUNBOOK.md`.

> TL;DR of the non-obvious ones: `--num-gpus-per-node` defaults to 8 (≠ our 4),
> gpt-oss `learnable softmax` (attention sinks) forbids `thd` and `context-parallel`
> in TE, `expandable_segments` breaks SGLang, and the 26k-seq OOM is solved by `TP=4`.

---

## A. Environment / plumbing

### A1. slime is a nested package
`slime/` (repo root) vs `slime/slime/` (import package). Files like `Sample` live at
`slime/slime/utils/types.py`. Don't look for `slime/utils/...`.

### A2. `load_function` has no colon syntax
`slime.utils.misc.load_function` splits on the **last dot** only. Pass fully-dotted
paths to every slime path arg (`--rollout-function-path a.b.c.fn`), never `a.b:fn`.

### A3. Keep EVERYTHING under /gscratch/zlab/lky04 (not /gscratch/zlab root)
- Container run with `--no-home`.
- Redirect ALL caches into the project: `HF_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `TORCH_HOME`, `WANDB_DIR/CACHE_DIR/CONFIG_DIR`,
  `APPTAINER_CACHEDIR/TMPDIR`.
- Reuse the prebuilt `apptainer/images/slime.sif` (copied into the project) — no build,
  no pip installs (slime + Megatron + SGLang + TE all live in the .sif).

---

## B. Ray cluster (multi-node, containerized)

### B1. Ray reports 7.82 TiB RAM you did not request
**Symptom:** `ray status` showed 7.82 TiB memory / 16 GPU.
**Cause:** `ray start` auto-detects each node's *physical* RAM (~1.95 TiB) and advertises
it. SLURM grant was only **512 GB/node**.
**Fix:** cap Ray to the grant: `ray start --memory 460e9 --object-store-memory 40e9`.
(Ray over-advertising could let it schedule past the cgroup -> SLURM OOM-kill.)

### B2. `/tmp/ray/ray_current_cluster` PermissionError on a worker
**Cause:** shared node `/tmp/ray` owned by another user/job.
**Fix:** give Ray a private temp dir: `ray start --temp-dir /tmp/rayttt_lky04`, and use
`RAY_ADDRESS=<head>:6379` for clients (`ray status`, `ray job submit`).

### B3. Containerized multi-node Ray lifecycle
Start head + workers as `apptainer exec ... ray start ... --block` backgrounded with
`nohup` (the `--block` keeps the container instance alive so the ray daemons persist).
Workers join via `--address <head_ip>:6379 --node-ip-address <worker_ip>`. apptainer uses
host networking, so ray ports are reachable across nodes.

### B4. `ray job submit` ran the entrypoint from $HOME
**Symptom:** `python3: can't open file '/mmfs1/home/lky04/examples/.../train_ttt.py'`.
**Cause:** the job's cwd was $HOME, not the repo; a relative entrypoint path failed.
**Fix:** use an **absolute** entrypoint: `python3 /root/slime/examples/ttt_discover/train_ttt.py`.

---

## C. SGLang multi-node (the big one)

### C1. SGLang engines TCPStore-timeout connecting cross-node (600s)
**Symptom:** `[c10d] The client socket has timed out ... connect to (10.64.77.125,15233)`;
engines on g3127/g3130 trying to reach g3125/g3128. All engines loaded ("ready to roll")
then hung 600s and died.
**Root cause:** slime's `--num-gpus-per-node` is a **separate arg defaulting to 8**, NOT
the same as `--actor-num-gpus-per-node 4`. With it at 8, slime thought each node had 8
GPUs, lumped **8 engines under one node IP** (spanning 2 physical 4-GPU nodes), and stamped
the base node's IP onto engines that actually ran on the neighbor -> cross-node dist that
times out.
**Fix:** `--num-gpus-per-node 4` (match physical GPUs/node). After this, engines place
4-per-node with their own IPs, 0 timeouts.
**Also needed (cross-node bootstrap NIC):** `NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=ens11f0np0`
(the fabric NIC carrying 10.64.77.x). And `--rollout-num-gpus-per-engine 1` (one gpt-oss
engine per card; this is what derives `sglang_tp_size`, NOT `--sglang-tp`).
**Diagnosis tip:** cross-node TCP works if a test connect is *refused* (fast) vs *timed out*
(dropped). Ours was refused on a random port -> fabric OK -> it was the IP-stamping bug.

### C2. (red herring) `--sglang-tp 1` is ignored
slime sets `sglang_tp_size = rollout_num_gpus_per_engine` (arguments.py). Control engine
size with `--rollout-num-gpus-per-engine`, not `--sglang-tp`.

---

## D. Megatron training init

### D1. Empty `--load` dir asserts
**Symptom:** `AssertionError: args.load=.../ckpts does not exist or is an empty directory`.
**Cause:** `--load` must point to a real checkpoint; on first run the dir is empty.
**Fix:** only pass `--load` when a real marker exists; else init from `--ref-load`:
`LOAD_ARG=""; [ -f "$CKPT/latest_checkpointed_iteration.txt" ] && LOAD_ARG="--load $CKPT"`.

### D2. Polluted ckpts dir -> "Unrecognized model ... config.json"
A crashed run left junk in `ckpts`; the old "load if dir non-empty" check wrongly fired.
**Fix:** require `latest_checkpointed_iteration.txt` (above) + clear stale `ckpts`.

---

## E. gpt-oss attention in Transformer Engine (the hard one)

gpt-oss uses **learnable softmax (attention sinks)** + SWA. With NVTE_DEBUG=2 we found TE
(2.10 / cuDNN 9.16) disables backends in these combos:

| inputs | result |
|---|---|
| `learnable + thd` (packed) | **NoBackend** |
| `learnable + context-parallel` | **NoBackend** |
| `learnable + bshd + no CP` | cuDNN **FusedAttention OK** |

### E1. `learnable + thd` -> No dot product attention backend
**Symptom:** `ValueError: No dot product attention backend is available`.
**Cause:** slime trains in **packed `thd`** format (varlen); TE disables Flash (learnable)
and Fused (learnable+thd) and Unfused (thd).
**Fix:** use **`--qkv-format bshd`** (padded) + drop `--use-dynamic-batch-size`
(bshd requires `--micro-batch-size` instead) + `--attention-backend fused`.
NB: do NOT set `NVTE_FLASH_ATTN/FUSED_ATTN` env manually — Megatron `--attention-backend`
manages them (manual env + `auto` asserts).

### E2. `expandable_segments` breaks SGLang
**Symptom (after trying the OOM-suggested allocator):**
`RuntimeError: TorchMemorySaver is disabled ... expandable_segments is not supported`.
**Cause:** SGLang's memory saver (colocate offload) is incompatible with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
**Fix:** do NOT set it (in colocate).

### E3. `learnable + context-parallel` -> No backend
**Symptom:** tried `--context-parallel-size 2` to split the 26k seq; TE:
`Disabling FusedAttention for context parallelism with softmax_type=learnable`.
**Cause:** CP is incompatible with attention sinks in this TE.
**Fix:** **cannot use CP** with gpt-oss here.

### E4. 26k-sequence OOM in bshd
**Symptom:** trained fine for ~5 steps, then a rollout produced a ~26k response; bshd pads
the microbatch to batch-max (~29k) -> full-attention layer -> `CUDA OOM: tried to allocate
10.84 GiB, 10.32 free` (just over; ~10 GiB lost to fragmentation).
**Cause:** can't use CP (E3) or expandable_segments (E2); per-layer attn/logits for 29k
is too big at TP=2.
**Fix:** **`--tensor-model-parallel-size 4`** — shards each layer's attention/logits 4-way,
the 10.84 GiB tensor -> ~5.4 GiB -> fits. Stable for 10+ steps.

---

## F. Operational

### F1. Monitor grabbed the wrong job
`ray job list | tail -1` is not time-ordered and returned stale/stopped jobs.
**Fix:** parse the newest submission id from `logs/train_submit.log`, or pass the job id
explicitly to the watcher.

### F2. Remote judge (yanagiorigami.uk) integration
- Endpoint is async: `POST /submit {"pid","lang","code"} -> {"sid"}`; poll `GET /result/{sid}`
  until `status=="done"` -> `{passed, result, score, scoreUnbounded, cases}`.
- Field names are `pid/lang/code` (not problem_id/language).
- Behind Cloudflare: default `Python-urllib` UA gets **403** -> send a browser User-Agent.
