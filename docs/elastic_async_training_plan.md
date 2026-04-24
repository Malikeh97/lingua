# Elastic Async Training Plan

## Core Idea

Uncoordinated, node-level training jobs across a **heterogeneous, disconnected fleet**:

- **L40S cluster** (4x L40S per node, has internet, PCIe interconnect)
- **H100 cluster** (4-8x H100 per node, no internet, NVLink interconnect)
- Clusters are **not directly connected** — only H100 login nodes can reach the internet

Each node:
1. Self-checks on launch, joins its cluster's pool
2. Holds a full model copy (optional SP within node)
3. Runs forward/backward independently
4. Accumulates gradients locally for K steps, then pushes to cluster aggregator
5. When the aggregator accumulates enough gradients (token-based threshold), it runs an optimizer step and broadcasts new weights
6. Dead nodes are simply absent — training continues without them

Cross-cluster sync happens through the H100 login node via **weight averaging** every few optimizer steps — frequent enough that no DiLoCo-style inner/outer optimizer split is needed.

---

## Two-Tier Architecture

### Overview

```
┌── L40S Cluster (internet) ───────────────────────────────────────────────┐
│                                                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                                    │
│  │ Node 0  │ │ Node 1  │ │ Node 2  │   (each node: 4x L40S)            │
│  │ sp=1    │ │ sp=1    │ │ sp=1    │   sp=1 → 4 independent workers    │
│  │ 4 wkrs  │ │ 4 wkrs  │ │ 4 wkrs  │   (PCIe too slow for ring attn)  │
│  └────┬────┘ └────┬────┘ └────┬────┘                                    │
│       │           │           │    push grads every K steps              │
│       └───────────┼───────────┘                                          │
│                   ▼                                                      │
│          ┌─────────────────┐                                             │
│          │ L40S Aggregator │  (optimizer, weights, token counter)        │
│          └────────┬────────┘                                             │
│                   │                                                      │
└───────────────────┼──────────────────────────────────────────────────────┘
                    │
                    │  weight averaging every 1-2 optimizer steps
                    │  via H100 login node (internet ↔ internal)
                    │  (~2 GB transfer in bf16 for 1B model)
                    │
┌───────────────────┼──────────────────────────────────────────────────────┐
│                   │                                                      │
│          ┌────────┴────────┐                                             │
│          │ H100 Aggregator │  (optimizer, weights, token counter)        │
│          └────────┬────────┘                                             │
│                   │                                                      │
│       ┌───────────┼───────────┐                                          │
│       │           │           │    push grads every K steps              │
│  ┌────┴────┐ ┌────┴────┐ ┌───┴─────┐                                    │
│  │ Node 0  │ │ Node 1  │ │ Node 2  │   (each node: 4-8x H100)         │
│  │ sp=4    │ │ sp=4    │ │ sp=4    │   sp=num_gpus → ring attn OK      │
│  │ 1 wkr   │ │ 1 wkr   │ │ 1 wkr   │   (NVLink is fast enough)       │
│  └─────────┘ └─────────┘ └─────────┘                                    │
│                                                                          │
└── H100 Cluster (no internet, login node bridges) ────────────────────────┘
```

### Tier 1: Within-Cluster (fast, async elastic)

Each cluster runs independently with its own aggregator:

- **Aggregator**: Holds canonical weights, gradient buffer, optimizer state (Adam m/v), token counter
- **Workers**: Accumulate gradients locally for `K` micro-batches, push to aggregator
- **Threshold**: Token-based. When accumulated token count >= threshold, aggregator averages gradients and runs `optimizer.step()`
- **Fault tolerance**: Dead workers are simply absent. Training continues with fewer gradient contributions.

### Tier 2: Across-Cluster (moderate frequency, weight averaging)

The two aggregators periodically sync through the H100 login node:

- **Mechanism**: Plain weight averaging `theta = (theta_L40S + theta_H100) / 2`
- **Frequency**: Every 1-2 optimizer steps — easily within gradient accumulation range
- **Transport**: H100 login node pulls weights from L40S aggregator (internet), pushes to H100 aggregator (internal network), and vice versa
- **Bandwidth**: Login node internet link (1-10 Gbps). For 1B model (2 GB bf16): 1.6-16 seconds per sync. Acceptable at the optimizer-step cadence.

**Why not DiLoCo?** At this sync frequency (~every 1-2 optimizer steps), the clusters barely diverge. Weight averaging is sufficient. DiLoCo's inner/outer optimizer split is only justified when H is in the hundreds (like Prime Intellect's H=500, forced by 127-935 Mbit/s intercontinental links). Our cross-cluster bandwidth is 10-100x better.

---

## Within-Node: SP as a Knob

Sequence parallelism is configured per node based on hardware:

```python
# Auto-configured per node
if "H100" in gpu_name:
    sp_size = num_gpus       # NVLink: ring attention is cheap
elif "L40S" in gpu_name:
    sp_size = 1              # PCIe: each GPU is independent worker
# Can always be overridden in config
```

| Hardware | Interconnect | SP recommendation | Workers per node |
|----------|-------------|-------------------|-----------------|
| 4x L40S | PCIe Gen4 (~32 GB/s) | `sp_size=1` | 4 independent |
| 4x H100 SXM | NVLink 4 (~900 GB/s) | `sp_size=4` | 1 (4-GPU SP group) |
| 8x H100 SXM | NVLink 4 | `sp_size=8` | 1 (8-GPU SP group) |

**Exception**: If sequences are short enough to fit on a single GPU, even H100 nodes can use `sp_size=1` for maximum throughput.

**Note on linear attention**: When using linear attention for cross-attention, the ring communication is just an AllReduce of a tiny state matrix `[n_heads, head_dim, head_dim_v]` (~512 KB for 32 heads, 128-dim). This is cheap even on PCIe, so SP could be viable on L40S for linear attention models. But for softmax cross-attention (full K/V rotation), PCIe is a real bottleneck. Keeping SP as a knob lets us make this choice per-experiment.

---

## Components

### 1. Cluster Aggregator

One per cluster. Runs on a CPU-capable node (doesn't need GPU). Ray actor or standalone process.

**State:**
```python
weights: Dict[str, Tensor]          # canonical model weights (bf16)
grad_buffer: Dict[str, Tensor]      # accumulated gradients
token_count: int = 0                # tokens processed in current accumulation
weight_version: int = 0             # monotonic version counter
optimizer: AdamW                    # from lingua/optim.py
optimizer_state: Dict               # Adam m (fp32), v (fp32)
token_threshold: int                # trigger optimizer step when reached
```

**On `receive_gradient(grad, num_tokens, from_version)`:**
```python
def receive_gradient(self, grad, num_tokens, from_version):
    # Optional: bounded staleness check
    for key in self.grad_buffer:
        self.grad_buffer[key] += grad[key]
    self.token_count += num_tokens

    if self.token_count >= self.token_threshold:
        # Average and step
        for key in self.grad_buffer:
            self.grad_buffer[key] /= (self.token_count / tokens_per_sample)
        self._apply_gradients()
        self._optimizer_step()
        self.weight_version += 1
        self._zero_buffers()
        self._maybe_checkpoint()
        self._maybe_cross_cluster_sync()
```

**Memory (aggregator):**

| Model size | Weights (bf16) | Adam m+v (fp32) | Grad buffer (fp32) | Total |
|-----------|---------------|-----------------|-------------------|-------|
| 1B | 2 GB | 8 GB | 4 GB | 14 GB |
| 3B | 6 GB | 24 GB | 12 GB | 42 GB |
| 7B | 14 GB | 56 GB | 28 GB | 98 GB |

Aggregator runs on CPU. CPU RAM is cheap and plentiful — even 7B is fine.

### 2. Worker

Runs on each GPU (sp_size=1) or each SP group (sp_size>1).

**Loop:**
```python
local_grad_buffer = {k: zeros_like(p) for k, p in model.named_parameters()}
local_token_count = 0

while True:
    # Pull new weights if available
    if aggregator.weight_version > local_version:
        model.load_state_dict(aggregator.get_weights())
        local_version = aggregator.weight_version

    # Accumulate locally for K micro-batches
    for _ in range(local_acc_steps):
        batch = data_pipeline.next()
        loss = model(batch, sp_group=sp_group)
        loss.backward()
        for k, p in model.named_parameters():
            if p.grad is not None:
                local_grad_buffer[k] += p.grad
        local_token_count += batch.num_tokens
        model.zero_grad()

    # Push accumulated gradient
    aggregator.receive_gradient(local_grad_buffer, local_token_count, local_version)
    zero_(local_grad_buffer)
    local_token_count = 0
```

### 3. Cross-Cluster Sync

Runs on H100 login node as a simple script/daemon:

```python
while True:
    # Fetch weights from both aggregators
    w_l40s = l40s_aggregator.get_weights()     # via internet
    w_h100 = h100_aggregator.get_weights()     # via internal network

    # Average
    w_avg = {k: (w_l40s[k] + w_h100[k]) / 2 for k in w_l40s}

    # Push back
    l40s_aggregator.set_weights(w_avg)
    h100_aggregator.set_weights(w_avg)

    # Wait for next sync point
    wait_for_next_optimizer_step()
```

More sophisticated options (weighted average by token contribution, one-leads-the-other) can be added later.

### 4. Pool Manager / Registry

One per cluster. Lightweight — tracks alive workers.

- Workers register on startup, heartbeat periodically
- Aggregator queries alive count for logging/monitoring
- Does NOT gate training — purely informational

---

## Gradient Accumulation: Two Levels

| Level | Where | Controlled by | Purpose | Typical value |
|-------|-------|--------------|---------|---------------|
| **Local** | Each worker | `local_acc_steps` | Reduce push frequency. Workers accumulate K micro-batches before pushing. | K = 8-32 |
| **Global** | Aggregator | `token_threshold` | Control effective batch size. Aggregator accumulates until enough tokens, then steps. | Depends on desired batch size |

**Example: 3 L40S nodes (12 workers), K=16, 4K tokens per micro-batch, threshold=1M tokens:**

```
Each push: 16 × 4K = 64K tokens
Pushes needed: 1M / 64K ≈ 16 pushes
With 12 workers: ~1.3 pushes per worker per optimizer step
→ Each worker does 16 × 1.3 ≈ 21 micro-batches per optimizer step
```

This is all standard gradient accumulation — no staleness, no weight divergence, no convergence concerns.

---

## Token-Based Threshold

The threshold counts **tokens processed**, not gradient pushes. This handles heterogeneous contributions correctly:

- H100 SP group (4 GPUs, long sequences): one push covers many tokens
- L40S single GPU (shorter sequences): one push covers fewer tokens
- Both contribute proportionally to the effective batch size

```python
# Worker pushes include token count
aggregator.receive_gradient(grad, num_tokens=batch.total_tokens, ...)

# Aggregator thresholds on tokens
if self.token_count >= self.token_threshold:
    self._optimizer_step()
```

---

## Node Lifecycle

```
[Launch] → [Self-check] → [Register w/ pool] → [Pull weights from aggregator]
                │                                        │
                ▼                                        ▼
          GPU health check                        [Training loop]
          NCCL intra-node test (if SP)                   │
          Memory check                                   ▼
                                              [Crash / Graceful exit]
                                                         │
                                                         ▼
                                              Aggregator notices missing
                                              heartbeat. No action needed.
```

**Joining mid-training**: Pull latest weights + version from aggregator. Start contributing immediately.

**Crash**: Aggregator detects missing heartbeat. Training continues — fewer gradient contributions per accumulation cycle, but threshold is token-based so the effective batch size stays correct.

---

## Checkpointing Strategy

**Aggregator-driven** (one per cluster):
1. Every N optimizer steps, save:
   - Model weights (plain `torch.save(state_dict)`)
   - Optimizer state (Adam m, v)
   - `weight_version`, `total_tokens_seen`, wall time
2. Keep last M checkpoints, delete older ones
3. Workers are stateless — they pull weights on restart

**Cross-cluster**: After weight averaging, the synced weights are checkpointed by both aggregators. Either checkpoint can recover the full training state.

**On aggregator crash**: Restart from latest checkpoint. Accumulated-but-unapplied gradients are lost (acceptable — it's at most one accumulation cycle worth of compute).

---

## Staleness

Within a cluster, staleness is minimal:

- A worker computes gradients on weights version V
- By the time it pushes (after K local steps), the aggregator may have advanced to V+1 or V+2
- With `local_acc_steps=16` and multiple workers, staleness is typically 0-2 versions

**Mitigation strategy**: Start by ignoring staleness. The large gradient accumulation naturally dampens any effect. Add bounded staleness rejection only if training diverges.

**Cross-cluster**: Both clusters run the same averaged weights after each sync. No staleness at the cross-cluster level as long as sync frequency is every 1-2 optimizer steps.

---

## Data Pipeline

Each cluster runs its own Ray data pipeline (independent):

- **L40S cluster**: Has internet access, can download/tokenize data directly
- **H100 cluster**: Data must be pre-staged to a filesystem accessible to compute nodes. Download and tokenize on L40S or H100 login node, write to shared storage.

Each pipeline uses a different random seed to avoid duplicate batches across clusters. Within a cluster, all workers pull from the same pipeline (via Ray actors) — deduplication is handled by the pipeline's Mixer/Packer.

---

## Open Design Decisions

### Model Size Constraint
Each node holds a full model copy. 4x L40S = 192 GB, 4x H100 = 320 GB.

| Model | Weights (bf16) | Activations (est.) | Fits on L40S node? | Fits on H100 node? |
|-------|---------------|--------------------|--------------------|---------------------|
| 1B | 2 GB | ~5 GB | Yes (7/192 GB) | Yes (7/320 GB) |
| 3B | 6 GB | ~12 GB | Yes (18/192 GB) | Yes (18/320 GB) |
| 7B | 14 GB | ~25 GB | Yes (39/192 GB) | Yes (39/320 GB) |

Note: Optimizer state lives on the aggregator (CPU), not on worker GPUs. Workers only hold model weights + activations.

### Gradient Compression (future optimization)
- **Start without compression** — local accumulation (K steps) already reduces push frequency
- **If bandwidth-limited**: int8 quantization (4x reduction) or TopK sparsification
- Within-cluster this is unlikely to matter; cross-cluster it may help

### Adaptive Threshold
Should `token_threshold` adapt to alive-worker count?
- Fixed threshold: effective batch size stays constant, training time per step varies with worker count
- Adaptive: `threshold = base_threshold * (alive_workers / expected_workers)` — keeps training speed constant but varies batch size
- Recommendation: fixed threshold (consistent training dynamics), accept variable speed

### Weight Averaging Strategy
Cross-cluster weight averaging options:
- **Equal average**: `(w_L + w_H) / 2` — simple, symmetric
- **Token-weighted**: Weight by tokens processed since last sync. If H100 cluster processed 3x more tokens, its weights get 3x the weight.
- **One-leads**: H100 cluster is authoritative (faster), L40S cluster pulls from it. L40S still contributes gradients to its local aggregator but defers to H100 weights at sync time.
- Recommendation: start with token-weighted average

---

## Comparison to Related Work

| System | Sync Model | Elasticity | Cross-cluster | Gradient Flow |
|--------|-----------|------------|---------------|---------------|
| **Standard FSDP/DDP** | Synchronous | None | N/A | AllReduce every step |
| **DiLoCo** | Periodic sync (H=500) | None | Designed for slow links | Inner optimizer + outer pseudo-gradient |
| **Hogwild!** | Lock-free async | N/A | N/A | Shared memory writes |
| **Parameter Server** | Async | Limited | Single cluster | Push grad / pull weights |
| **This proposal** | Async + threshold | Full elastic | Weight averaging via login node | Local acc → push to cluster aggregator → cross-cluster avg |

Key differentiator: Two-tier design that matches sync frequency to available bandwidth at each level. Fast intra-cluster aggregation (datacenter network), moderate cross-cluster averaging (login node internet link). No DiLoCo complexity needed because cross-cluster bandwidth is sufficient for frequent syncing.

---

## Reuse Analysis

> **Key distinction**: `lingua/` is the **upstream Facebook Research library** (forked, mostly untouched). `addons/` is **our own code** built on top. These have very different implications for reuse — we own `addons/` and can freely modify it, while `lingua/` is a dependency we import from.

---

### addons/ (ours) — Component-by-Component

| Component | File(s) | Reuse? | Assessment |
|-----------|---------|--------|------------|
| **Models** | `addons/models/encoder_decoder.py`, `decoder.py` | **Yes, directly** | Clean. `sp_group` is an optional param passed through, no distributed logic baked into the model itself. This is the highest-value reuse — all the research code (cross-attention adapters, packed sequences, encoder-decoder architecture) transfers as-is. |
| **Attention (local)** | `addons/models/attention.py` | **Yes, directly** | Flash attention, all 10+ linear attention variants (gla, delta_rule, kda, etc.) work with `sp_group=None`. The kernel integrations (FLA, flash-attn) are the hard work here. |
| **Attention (ring)** | `addons/models/attention.py` | **Yes, within-node** | Ring attention is already scoped to a process group. For intra-node SP, pass the local SP group — works unchanged. Not used cross-node in the elastic design. |
| **SP groups** | `addons/distributed.py` | **Yes, within-node** | `get_sp_group()`, `get_dp_group()`, `ring_send_recv()` — all 113 lines. Useful for intra-node SP setup. The groups are created from a process group, so they work with a node-local `init_process_group`. No changes needed. |
| **Ray data pipeline** | `addons/data/ray_pipeline.py` | **Yes, directly** | Zero rank awareness. Each node can independently create a pipeline or share one via Ray. This is the strongest infrastructure reuse — the whole actor-based pipeline (DatasetReader -> Mixer -> Packer -> Coordinator) just works. |
| **Collation / packing** | `addons/data/collate.py` | **Partially** | `TokenizedBatch`, `PackedSequences`, packing logic — all reusable. `broadcast_batch()` / `shard_batch()` are SP-specific; useful within-node if doing SP, irrelevant cross-node. |
| **Data config** | `addons/data/config.py` | **Yes, directly** | Plain dataclass. |
| **Trainer** | `addons/trainer.py` | **Partially** | The forward/backward/gradient-clip inner logic is fine. But the step structure assumes synchronous training: it calls `optimizer.step()`, manages grad accumulation locally, etc. In elastic, workers don't step — they extract and push gradients. Would need to fork this into an `ElasticWorkerTrainer` that replaces the optimizer step with gradient push. Maybe 50% of the logic carries over. |
| **Tasks / eval** | `addons/tasks/` | **Yes, directly** | Evaluation tasks (HotpotQA, Squad, LongBench, etc.) are independent of training infra. Run on any model instance. |

**addons/ verdict**: The research code (models, attention, data pipeline, tasks) is **genuinely reusable** and represents the majority of the codebase's value. The distributed/training glue (`addons/distributed.py`, `addons/trainer.py`) is small and partially reusable — the parts that don't apply are easy to replace.

---

### lingua/ (upstream FB) — Component-by-Component

| Component | File(s) | Reuse? | Assessment |
|-----------|---------|--------|------------|
| **Optimizer** | `lingua/optim.py` | **Yes, directly** | Pure AdamW + LR schedules (linear, cosine, warmup-stable-decay). Zero distributed assumptions. Runs on the aggregator. ~150 lines, all clean. |
| **Metrics** | `lingua/metrics.py` | **Mostly** | `MetricLogger` (JSONL + wandb), `GPUMemoryMonitor` — both work. `MetricLogger.__init__` calls `get_is_master()` to gate wandb — needs a small tweak for elastic (pass a flag instead of reading rank). |
| **Logger** | `lingua/logger.py` | **Mostly** | Log formatter with rank prefix. Caches rank at init time from env vars. Works fine if `dist.is_initialized()` on the node; just shows local rank. |
| **Tokenizer** | `lingua/tokenizer.py` | **Yes, directly** | Pure tokenizer wrapper. No distributed coupling. |
| **Distributed** | `lingua/distributed.py` | **Hollow** | ~500 lines. The bulk is: `init_device_mesh()` (rigid topology), `parallelize_model()` (FSDP `fully_shard()` wrapping), `DistributedArgs` (fixed dp/tp/fsdp config). Usable bits: `setup_env()` (~10 lines of NCCL env var tuning), `init_signal_handler()` / `requeue_slurm_job()` (~20 lines for SLURM preemption). That's ~30 lines out of 500. |
| **Checkpointing** | `lingua/checkpoint.py` | **Hollow** | ~350 lines built around PyTorch DCP. `save()` calls `dcp.save()` — requires all ranks to participate. For elastic: the aggregator does `torch.save(state_dict, path)`. The "save every N steps, keep M" scheduling is ~15 lines of logic worth copying. |
| **Profiling** | `lingua/profiling.py` | **Mostly** | xformers profiler wrappers. Has a `dist.barrier()` call that would need guarding. Otherwise fine. |
| **Data** | `lingua/data.py` | **No** | Upstream data loading. We use `addons/data/` (Ray pipeline) instead. |
| **Transformer** | `lingua/transformer.py` | **No** | Upstream LLaMA-style model. We use `addons/models/` instead. |

**lingua/ verdict**: `optim.py` and `metrics.py` are genuinely useful. `distributed.py` and `checkpoint.py` solve the wrong problem (synchronous FSDP with fixed topology).

---

### Summary: What's Real vs What's Wrapping

```
GENUINELY REUSED (saves real work)           WHO OWNS IT
-------------------------------------------------------------
addons/models/*          models, attention       ours
addons/data/*            ray pipeline, collate   ours
addons/distributed.py    SP groups, ring ops     ours  (within-node only)
addons/tasks/*           evaluation              ours
lingua/optim.py          optimizer + LR          upstream, used as-is
lingua/metrics.py        logging + wandb         upstream, minor tweak

PARTIALLY REUSED (fork/adapt ~50%)
-------------------------------------------------------------
addons/trainer.py        inner fwd/bwd loop      ours, fork into elastic worker

HOLLOW / NOT USED
-------------------------------------------------------------
lingua/distributed.py    FSDP mesh, fully_shard   upstream, wrong paradigm
lingua/checkpoint.py     DCP coordinated saves    upstream, wrong paradigm
lingua/data.py           upstream data loading     upstream, replaced by addons/
lingua/transformer.py    upstream LLaMA model      upstream, replaced by addons/
```

---

## What We'd Write New

All new code lives in `addons/elastic/`:

1. **`elastic/aggregator.py`** (~200-300 lines)
   - Gradient buffer management, weight versioning, token-based threshold
   - Optimizer step (imports `lingua.optim.build_optimizer`)
   - Weight serving (Ray actor with get/set)
   - Checkpoint: plain `torch.save` / `torch.load` + "keep last M" logic

2. **`elastic/worker.py`** (~150-200 lines)
   - Self-check (GPU health, NCCL intra-node smoke test)
   - Weight pull / version check from aggregator
   - Training loop: local accumulation for K steps -> push to aggregator
   - Reuses: model from `addons/models`, data from `addons/data`, SP from `addons/distributed`

3. **`elastic/cross_cluster.py`** (~50-80 lines)
   - Weight averaging daemon that runs on H100 login node
   - Pulls from both aggregators, averages, pushes back
   - Triggered every N optimizer steps or on a timer

4. **`elastic/pool.py`** (~50-100 lines)
   - Node registry (Ray named actor)
   - Heartbeat monitoring, alive-node count

5. **`elastic/launch.py`** (~50 lines)
   - Per-node launcher (replaces `torchrun`)
   - Intra-node `init_process_group` for SP
   - Auto-detect hardware -> set sp_size
   - Self-check -> register -> start worker

**Total new code: ~500-730 lines**

### Architecture: Full Picture

```
┌──────────────────────────────────────────────────────────────┐
│                 L40S CLUSTER AGGREGATOR (NEW)                 │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ Gradient Buffer   │  │ Optimizer  <- lingua/optim.py     │  │
│  │ Token Counter     │  │ Metrics    <- lingua/metrics.py   │  │
│  │ Weight Versioning │  │ Checkpoint: torch.save (trivial)  │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────┘
         push grads (Ray)      │       pull weights (Ray)
    ┌──────────────────────────┼────────────────────────┐
    │              │           │           │             │
┌───┴───┐    ┌────┴──┐   ┌───┴───┐   ┌───┴───┐    ┌───┴───┐    L40S: sp=1
│GPU 0  │    │GPU 1  │   │GPU 2  │   │GPU 3  │    │GPU ..│    4 workers/node
│worker │    │worker │   │worker │   │worker │    │worker│
└───────┘    └───────┘   └───────┘   └───────┘    └───────┘
  Model <- addons/models (ours)    Data <- addons/data (ours)

                              │
                    weight avg via login node              <- NEW (~50 lines)
                              │

┌─────────────────────────────┴────────────────────────────────┐
│                 H100 CLUSTER AGGREGATOR (NEW)                 │
│  (same code as L40S aggregator, different config)             │
└──────────────────────────────┬───────────────────────────────┘
         push grads (Ray)      │       pull weights (Ray)
    ┌──────────────────────────┼──────────────────┐
    │                          │                  │
┌───┴──────────────┐   ┌──────┴───────────┐  ┌───┴──────────────┐    H100: sp=4
│[GPU0-SP-GPU1     │   │[GPU0-SP-GPU1     │  │[GPU0-SP-GPU1     │    1 worker/node
│     -SP-GPU2     │   │     -SP-GPU2     │  │     -SP-GPU2     │    (ring attn)
│     -SP-GPU3]    │   │     -SP-GPU3]    │  │     -SP-GPU3]    │
│   1 worker       │   │   1 worker       │  │   1 worker       │
└──────────────────┘   └──────────────────┘  └──────────────────┘
  Ring attn <- addons/models/attention.py (ours)
  SP groups <- addons/distributed.py (ours)
```

### Bottom Line

**addons/ is almost entirely reusable** — models, attention kernels, data pipeline, eval tasks. This is the research code and it transfers cleanly.

**lingua/ contributes `optim.py` and `metrics.py`**. The rest (`distributed.py`, `checkpoint.py`) solves the wrong problem.

**~500-730 lines of new code** for the elastic layer (aggregator, worker, cross-cluster sync, pool, launcher) on top of the existing foundation.
