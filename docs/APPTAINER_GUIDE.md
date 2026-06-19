# Running Docker-based ML code with Apptainer/Singularity (HPC, no root)

A practical guide to running container images that are *published as Docker images*
(e.g. `slimerl/slime`, `nvcr.io/...`, `vllm/...`) on a shared HPC cluster where you
**don't have root and can't use Docker** — using **Apptainer** (formerly Singularity).
Everything here is battle-tested from deploying TTT-on-slime on the UW Hyak cluster
(SLURM + H200 nodes). Commands say `apptainer`; on older clusters substitute `singularity`.

> TL;DR of the hard-won rules:
> 1. **Build/convert images on a compute node, never the login node** (the login
>    watchdog kills heavy processes).
> 2. **Point every cache/temp dir into your own project space** (`$PROJ/...`), never
>    `$HOME`, never a shared `/tmp` — *except Ray's `--temp-dir`, which must be node-local.*
> 3. **The image version must match the code you bind into it** (megatron-core / sglang /
>    transformers are tightly coupled to the framework version).
> 4. **`--nv` for GPUs, `--no-home` for isolation, `--bind` to inject data and your own code.**

---

## 0. Mental model

A Docker image is a stack of layers. Apptainer converts it into a **single immutable
`.sif` file** you can `exec` into. Unlike Docker:

- No daemon, no root. You run as **yourself**; UID is preserved into the container.
- The container filesystem is **read-only**. You make data visible with **bind mounts**.
- GPUs are exposed with **`--nv`** (binds the host driver/libraries in).

So the workflow is: **(1) get a `.sif`**, then **(2) `apptainer exec` it with the right
binds + env**, optionally **(3) overlaying your own code** over the image's copy.

---

## 1. Getting the image: `docker://` → `.sif`

```bash
apptainer pull <out>.sif docker://<org>/<image>:<tag>
```

This **downloads the layers** and then **converts OCI → SIF** (extract rootfs →
`mksquashfs`). Two phases, and the second is the slow/heavy one.

### Gotchas that will bite you

**a) Do the conversion on a compute node, not the login node.**
The OCI→SIF step extracts a full rootfs (hundreds of thousands of small files: Python +
CUDA + torch + the framework) and then squashes it. On a login node this is CPU/RAM/IO
heavy and the cluster's **login-node watchdog will SIGKILL it mid-extraction** (you'll see
the process just vanish, `.sif` never appears). Run it inside a small batch job:

```bash
#SBATCH -A <acct> -p <gpu-partition> -N1 --gpus-per-node=1 -c 32 --mem=256G -t 2:00:00
export APPTAINER_CACHEDIR=$PROJ/cache/apptainer   # downloaded blobs (reused across pulls)
export APPTAINER_TMPDIR=$PROJ/cache/apt_tmp       # extraction + squashfs scratch
mkdir -p "$APPTAINER_TMPDIR"
cd $PROJ/images && apptainer pull --force img.sif docker://org/image:tag
```

**b) Cache + temp dirs must live in *your* space.** By default Apptainer uses `$HOME` and
`/tmp`. On a shared cluster you want everything under your project dir
(`APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`). The blobs in `CACHEDIR` are **reused** on a
re-pull, so if a conversion dies you don't re-download 20+ GB.

**c) GPFS/Lustre is slow for this.** Network parallel filesystems handle the
"millions of tiny files" of an extracted rootfs poorly (metadata round-trips per file).
A 24 GB image can take ~45–60 min to convert on GPFS vs minutes on node-local NVMe. If
your site lets you use node-local scratch for *temporary* build space it's much faster —
but if policy says "stay in my project dir," just accept the slowness; it's a one-time cost.

**d) `.sif` appears only at the very end.** The final `Creating SIF file...` (mksquashfs)
writes the output in one shot. Don't mistake a long-silent extraction for a hang — check
that the process is alive and `APPTAINER_TMPDIR` is growing.

**e) Pin the tag to match your code.** `:latest` tracks upstream `main`; a versioned tag
(e.g. `:v0.3.0`) matches that release. If you **bind your own copy of the framework** into
the container (see §3), the image's bundled libs (megatron-core, sglang, transformers,
flash-attn, torch) must be compatible with *your* code version. A skew shows up as
cryptic runtime errors (unmapped weight names, missing model classes, ABI errors), **not**
as a clean "version mismatch." When in doubt, use the image built from the same commit/tag
as your code.

---

## 2. Running: `apptainer exec`

```bash
apptainer exec \
  --nv \                         # expose NVIDIA GPUs (host driver/libs)
  --no-home \                    # don't bind $HOME (clean, reproducible env)
  --bind /scratch:/scratch \     # make host paths visible inside
  --bind $PROJ/data:/data \
  --env HF_HOME=/cache/hf \      # redirect every cache into bound space (see §4)
  img.sif \
  python train.py ...
```

- **`--nv`** is required for CUDA. Without it you get `libcuda.so.1: cannot open shared
  object file` (the host driver isn't mounted). Note: this means GPU imports only work on
  a node that actually has a GPU — a login-node `apptainer exec ... python -c "import torch"`
  may fail on CUDA bits even though the image is fine.
- **`--no-home`** avoids leaking your `$HOME` dotfiles/conda into the container. Highly
  recommended for reproducibility.
- **`--bind A:B`** mounts host path `A` at container path `B`. Repeatable. This is how all
  your data/models/outputs get in and out (the rootfs itself is read-only).
- **`--writable-tmpfs`** gives a small writable overlay if some tool insists on writing
  inside the image; prefer binding a real dir instead.

### Quick file/version checks without a GPU
You can inspect an image on the login node *as long as you don't import CUDA*:
```bash
apptainer exec img.sif python -c "import transformers, importlib.util, os; \
  print(transformers.__version__); \
  print(os.path.dirname(importlib.util.find_spec('sglang').origin))"
```
`importlib.util.find_spec(...)` locates a package **without importing** it (so no CUDA).
Listing files inside the `.sif` (`apptainer exec img.sif ls .../models/`) still has to
cold-read the squashfs off GPFS, so it can be slow the first time.

---

## 3. Overlaying your own code over the image's copy

Images usually ship the framework at a fixed path (e.g. `/root/slime`,
`/workspace/vllm`). To run **your modified version** without rebuilding the image, bind
your working tree over that path:

```bash
apptainer exec --nv --no-home \
  --bind $PROJ/slime:/root/slime \        # your code shadows the image's /root/slime
  --env PYTHONPATH=/root/slime:/root/Megatron-LM \
  img.sif python /root/slime/train.py ...
```

Caveats:
- Your code now runs against the **image's** compiled libs (megatron-core, sglang kernels,
  flash-attn). They must be compatible — see §1e.
- For *pure* operations that ship with the framework (e.g. a checkpoint converter), it's
  often safer to use the image's **built-in** copy (don't bind your tree) so the converter
  and the libs are guaranteed consistent. Bind your tree only where you need your changes
  (e.g. training with a custom plugin).

---

## 4. Redirect every cache (so nothing escapes your project dir)

ML stacks scatter caches across `$HOME` and `/tmp`. On a shared cluster, redirect them all
into your project space. Pass these on **every** `apptainer exec` — including one-off
verification commands, or torchinductor/triton silently create dirs like
`/tmp/torchinductor_$USER`:

```bash
--env HF_HOME=$PROJ/hf_cache \
--env HUGGINGFACE_HUB_CACHE=$PROJ/hf_cache \
--env XDG_CACHE_HOME=$PROJ/cache \
--env TRITON_CACHE_DIR=$PROJ/cache/triton \
--env TORCHINDUCTOR_CACHE_DIR=$PROJ/cache/inductor \
--env TORCH_HOME=$PROJ/cache/torch \
--env TOKENIZERS_PARALLELISM=false
```

---

## 5. Multi-node: a containerized Ray/torch cluster over SLURM

Running a multi-node job (Ray head + workers, or torchrun) where every process is inside a
container adds networking wrinkles:

- **Pin the fabric NIC for collectives.** Auto-detection often picks the wrong interface
  and you get TCPStore/NCCL timeouts that look like hangs. Set:
  ```bash
  --env NCCL_SOCKET_IFNAME=<fabric-iface> --env GLOO_SOCKET_IFNAME=<fabric-iface>
  ```
  Find the interface with `ip -o -4 addr` on a compute node (e.g. `ens11f0np0`).
- **Resolve head IP to the fabric subnet**, not a link-local/IPv6 address. Under SLURM,
  derive nodes from `$SLURM_JOB_NODELIST` (`scontrol show hostnames`) and pick the IPv4 on
  the right subnet: `getent ahostsv4 $host | awk '{print $1}' | grep -m1 '^10[.]'`.
  (A stray `fe80::` link-local address from `getent hosts` will silently break ssh/ray.)
- **Match the engine size to physical GPUs/node.** If an inference engine is told there are
  8 GPUs/node but a node has 4, it tries to span nodes and the rendezvous times out. Set
  the framework's "GPUs per node" to the real number.
- **Cap container memory to your SLURM grant.** Ray (and others) auto-detect *physical*
  RAM and may try to grab far more than your cgroup allows. Pass explicit caps
  (`ray start --memory ... --object-store-memory ...`).
- **Ray's `--temp-dir` MUST be node-local.** Ray puts **Unix domain sockets** there, and
  network filesystems (GPFS/NFS) don't support sockets — so this is the one thing you
  cannot keep in your GPFS project dir. Use a node-local path like `/tmp/ray_$USER` and
  clean it up on exit (`ray stop` / job teardown). Flag this exception to yourself; it's the
  rare legitimate use of node-local `/tmp`.

Pattern: `srun`/sbatch allocates N nodes → start a Ray head on node 0 (fabric IP) → start
workers on the rest, **all via `apptainer exec` with identical binds/env** → submit your
job to the head. Keep the bind/env definition in one sourced file so head, workers, and the
job command are guaranteed identical.

---

## 6. Secrets & hygiene

- **Never bake secrets into the image or commit them.** Keep API keys in a `chmod 600`
  file (e.g. `wandb_secret.sh`) that you `source` at runtime and pass through as
  `--env WANDB_API_KEY=$WANDB_API_KEY`. Add that file to `.gitignore`.
- **Clean up only your own artifacts** on shared nodes (`find /tmp -maxdepth 1 -user $USER`),
  never touch others' files.
- **Don't put data/outputs in the image.** Bind them. The `.sif` should be reproducible
  from `docker://` + your bound code.

---

## 7. Cheat sheet

```bash
# Build (in a batch job, caches/tmp in your project dir):
APPTAINER_CACHEDIR=$PROJ/cache/apptainer APPTAINER_TMPDIR=$PROJ/cache/apt_tmp \
  apptainer pull --force img.sif docker://org/image:tag

# Inspect without GPU (login node ok):
apptainer exec img.sif python -c "import transformers; print(transformers.__version__)"

# Run training (GPU node), your code overlaid, caches redirected:
apptainer exec --nv --no-home \
  --bind /scratch:/scratch --bind $PROJ/slime:/root/slime --bind $PROJ/data:/data \
  --env PYTHONPATH=/root/slime:/root/Megatron-LM \
  --env HF_HOME=$PROJ/hf_cache --env XDG_CACHE_HOME=$PROJ/cache \
  --env NCCL_SOCKET_IFNAME=<iface> --env GLOO_SOCKET_IFNAME=<iface> \
  img.sif python /root/slime/train.py ...
```

### Symptom → cause map
| Symptom | Likely cause |
|---|---|
| `apptainer pull` process vanishes, no `.sif` | login-node watchdog killed the conversion → use a compute node |
| Pull "hangs" for a long time | GPFS small-file IO during extract/squashfs → normal, or use node-local tmp |
| `libcuda.so.1: cannot open shared object file` | missing `--nv`, or running GPU code on a CPU/login node |
| Cross-node NCCL/TCPStore timeout | wrong `*_SOCKET_IFNAME`, or engine GPUs/node ≠ physical |
| ssh/ray gets garbage host address | resolved an IPv6/link-local addr → force IPv4 fabric subnet |
| Ray won't start on network FS | `--temp-dir` on GPFS/NFS (sockets unsupported) → node-local `/tmp` |
| Unmapped weight names / missing model class at runtime | image libs ≠ your bound code version → match the image tag to your code |
| `/tmp/torchinductor_$USER` etc. appearing | cache env vars not passed on that `apptainer exec` |
```
