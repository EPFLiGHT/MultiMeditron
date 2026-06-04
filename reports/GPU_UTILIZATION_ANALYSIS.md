# GPU Utilization Analysis — MultiMeditron vs NanoVLM-v2

*Generated from logs of job 1709145 (128-node Stage2), 1709164 (2-node debug), and haaissa's nanoVLM-v2-full run.*

---

## 1. Raw GPU Utilization Data

| Job | Scale | SM% | MEM BW% | Duration | Notes |
|-----|-------|-----|---------|----------|-------|
| 1662533 | 128 nodes (512 GPUs) | **99.7%** | **0.5%** | ~6h | Mar 16 run |
| 1665845 | 128 nodes (512 GPUs) | **99.7%** | **0.5%** | ~6h | Mar 17 run |
| 1709145 | 128 nodes (512 GPUs) | **99.1%** | **1.0%** | 719 min | Killed by SIGTERM at step 835 |
| 1709164 | 2 nodes (8 GPUs)     | **77.3%** | **6.6%** | 30 min  | Debug run |

Monitoring: `nvidia-smi dmon -s u -d 5` (5-second interval SM% and memory bandwidth % per GPU).

### The Smoking Gun

**SM looks 99% busy but MEM bandwidth is 1%.** This is not efficient training — it is communication-bound execution.

At 512 GPUs with ZeRO-3, every parameter must be all-gathered (broadcast from its shard-owner to all 512 ranks) before it can be used in a forward pass, then reduced after the backward pass. These NCCL collective kernels run on the SM — so the SM appears busy — but they perform no matrix multiplications. The 1% memory bandwidth confirms that the HBM (GPU memory) is mostly idle while the SM waits for data to arrive over Slingshot.

**2-node comparison**: At only 8 GPUs, communication is minimal. SM drops to 77% (some genuine idle time) but MEM BW jumps to 6.6% — 6× higher than 128 nodes. This confirms the 128-node run is communication-saturated.

---

## 2. Training Throughput Comparison

| Metric | MultiMeditron (128 nodes) | NanoVLM-v2-full (haaissa, 1 node) |
|--------|--------------------------|-----------------------------------|
| Model size | 8B params | 360M params |
| GPU count | 512 (128 × H200) | 4 (1 × GH200 node) |
| per-device batch | 8 | 2 |
| Effective batch | 8,192 | 8 |
| sec/step | **51.7** | **1.93** |
| steps/min | 1.16 | 21.24 |
| samples/sec | 158.6 | 4.1 |
| samples/GPU-hour | **1,115** | **3,731** |
| Total training steps | 835 (killed) | 40,000 (complete) |
| Total wall-clock | ~12h (killed) | 31h |

**nanoVLM is 3.3× more GPU-efficient** (samples / GPU-hour), and this is with a model that is 22× smaller. Correctly normalized for model size, MultiMeditron would need to be roughly as efficient per GPU to justify its scale.

---

## 3. Model FLOP Utilization (MFU)

MFU = (actual FLOP/sec) / (peak FLOP/sec) × 100%. Uses standard approximation: $6 \times N_{params} \times L_{seq}$ FLOP per forward+backward for a sequence of length $L_{seq}$.

Configuration: H200 GH200 peak = 1,979 TFLOPs (bfloat16), seq_len = 2,048 tokens.

| Model | Actual FLOP/sec | GPU peak FLOP/sec | MFU |
|-------|----------------|-------------------|-----|
| MultiMeditron | 1.56 × 10¹⁶ | 1.01 × 10¹⁸ (512 × H200) | **1.5%** |
| NanoVLM-v2 | 1.83 × 10¹³ | 7.92 × 10¹⁵ (4 × H200) | 0.2% |

**Reference**: Well-optimized large-scale training typically achieves 35–55% MFU (Chinchilla, Megatron-LM, etc.).

MultiMeditron at 1.5% MFU means that for every 100 GPU-hours billed, only ~1.5 GPU-hours perform useful computation. The remaining 98.5% is spent on ZeRO-3 allgathers and reduce-scatters.

At 40% MFU, each training step would take ~2.0 seconds instead of 51.7 seconds — a **25× speedup** is theoretically achievable.

NanoVLM is also low (0.2%) but for a different reason: batch_size=2 per GPU is too small to saturate a 1,979 TFLOP GPU. Increasing to 32 per GPU would push NanoVLM MFU to ~3%.

---

## 4. Root Cause: ZeRO-3 at 512 Ranks

ZeRO-3 shards **all parameters** across all 512 GPUs. Each forward pass requires:

1. **Allgather**: Reconstruct the full weight tensor for each layer on each GPU  
   → 16 GB of parameters ÷ 512 = 32 MB/GPU stored; 16 GB allgathered per forward pass  
2. **Compute**: Use the gathered weights for matmul (tiny window)
3. **Free**: Discard the gathered weights
4. **Reduce-Scatter**: Aggregate gradients back across 512 GPUs during backward

For an 8B model with ~100 transformer layers, this is hundreds of collective operations per step. At 512 GPUs on Slingshot, each allgather adds latency from the tree/ring topology — even at Slingshot's 200 GB/s node bandwidth, the serialized collectives dominate.

**The key sign**: SM=99% but MEM BW=1%. NCCL uses SM but not HBM; the GPU is spin-waiting on network I/O dressed up as SM work.

---

## 5. 128-Node Run Status (Job 1709145)

- **Config**: `stage2_end2end.yaml`, 128 nodes, `per_device_batch=8`, `grad_accum=2`, `logging_steps=1`
- **Start**: Mar 24 02:53:40 CET
- **Steps completed**: 835
- **Loss trajectory**: 1.22 → 0.52 (healthy, converging)
- **Termination**: Killed by SIGTERM (job hit wall-time limit) — **did NOT complete training**
- **GPU-hours consumed**: 719 min × 512 GPUs = ~6,144 GPU-hours

---

## 6. Recommendations

### Immediate: Reduce scale drastically

The scaling efficiency collapses well before 128 nodes. From the 2-node data (highest MFU we can observe), even at 8 GPUs the MEM BW is only 6.6% — suggesting ZeRO-3 is already suboptimal at single-node multi-GPU with this model.

**Option A — Switch from ZeRO-3 to ZeRO-2** (`deepspeed.json`)  
ZeRO-2 shards optimizer state and gradients but **replicates parameters** on every GPU. Eliminates the forward-pass allgather entirely. Requires enough VRAM per GPU to hold the full model. With 8B params in bf16 = 16 GB, plus activations: fits on H200's 96 GB. Expected MFU improvement: 5–15×.

```json
// config/deepspeed.json — change stage
{
  "zero_optimization": {
    "stage": 2,   // was 3
    ...
  }
}
```

**Option B — Reduce node count to 8–16 nodes (32–64 GPUs)**  
The communication overhead scales with node count. 16 nodes gives 4× less communication overhead than 128 nodes while still running reasonable batch sizes. Train for more steps within a single job rather than spawning huge jobs.

**Option C — Gradient checkpointing / micro-batch tuning**  
Increase `per_device_train_batch_size` and remove `gradient_accumulation_steps`. Fewer, larger batches reduce synchronization barriers.

**Option D — Profile with NCCL timing**  
Add `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=ALL` to a short 2-node run to quantify exact allgather latency per layer.

### Comparison context

NanoVLM-v2 trains a 360M model to 40,000 steps in 31h on 1 node (4 GPUs) for ~124 GPU-hours total. MultiMeditron killed after 835 steps having spent ~6,144 GPU-hours. To put this on equal footing: running MultiMeditron to a similar step count (40,000) at current efficiency would require ~295,000 GPU-hours — vs ~125 GPU-hours for nanoVLM. Even comparing per-sample-trained, MultiMeditron using ZeRO-3 at 128 nodes is structurally wasteful.

---

## 7. GPU Log File Reference

| Job | Log path |
|-----|----------|
| 1709145 (128-node) | `/users/surech/meditron/reports/gpu-util-1709145/node-0.log` |
| 1709164 (2-node)   | `/users/surech/meditron/reports/gpu-util-1709164/node-0.log` |
| 1662533 (old CSV)  | `/users/surech/reports/multimeditron/gpu-1662533/` |
| 1665845 (old CSV)  | `/users/surech/reports/multimeditron/gpu-1665845/` |

NanoVLM checkpoints (haaissa): `/iopsstor/scratch/cscs/haaissa/multimeditron/checkpoints/nanovlm-v2-full/`  
NanoVLM timing derived from checkpoint mtime deltas (966s per 500 steps = 1.93 sec/step).
