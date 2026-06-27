# MultiMeditron — Possible Optimizations

> Analysis date: 2026-05-08  
> Based on smoke-test benchmarks (see `reports/smoketest_summary.md`) and code review of the training pipeline.

---

## Part 1 — Data Pipeline Bottlenecks

The GPU SM utilization across all benchmarks was 8–25% average, meaning the GPU is idle 75–92% of the time. Three root causes were identified by reading the code.

### 1.1 `copy.deepcopy(tokenizer)` on every single batch — highest priority

**File**: [`src/multimeditron/model/data_loader.py:102`](../src/multimeditron/model/data_loader.py), [`src/multimeditron/model/prompt_tokenizers.py:37`](../src/multimeditron/model/prompt_tokenizers.py)

`DataCollatorForMultimodal.torch_call()` is the collate function — it runs on **every batch**. Inside it, a new `SamplePreprocessor` is instantiated, which creates a new `PromptTokenizer`, which calls:

```python
self.tokenizer = copy.deepcopy(tokenizer)  # deep-copies the full vocabulary every batch
```

A tokenizer deep-copy includes the full vocabulary, merge rules, special token maps, and regex objects. For a model like SmolLM2-360M this is already measurable; for Meditron3-8B with an extended medical vocabulary it is significantly more expensive.

**Fix**: Move `SamplePreprocessor` construction into `DataCollatorForMultimodal.__post_init__` (or `__init__`), so it is created once at startup and reused across all batches. The tokenizer is not mutated during collation, so sharing it is safe.

```python
# In DataCollatorForMultimodal.__post_init__:
def __post_init__(self):
    self._sample_preprocessor = SamplePreprocessor(
        tokenizer=self.tokenizer,
        chat_template=self.chat_template,
        modality_processors=self.modality_processors,
        attachment_token=self.attachment_token,
        max_length=self.max_length,
        truncation=self.truncation,
    )

# In torch_call, replace the construction with:
modality_preprocessor = self._sample_preprocessor
```

---

### 1.2 Small batch size: structural GPU underutilization

**Config**: `per_device_train_batch_size: 8` (stage2 production), `2` (smoke tests)

At batch size 2, each forward pass launches GEMMs with tiny M-dimensions against 96 GB of HBM3. The GPU's Tensor Cores are never saturated — kernel launch overhead and memory latency dominate. This is a fundamental limit: no data pipeline fix will raise SM% significantly at batch=2.

**Fix options**:
- Increase `per_device_train_batch_size` as far as memory allows (see Part 2 for memory budget per strategy)
- Increase `gradient_accumulation_steps` in proportion to keep effective batch size constant
- For the small NanoVLM models (SmolLM2-360M), batch=32–64 should be well within budget

---

### 1.3 Online image decode and CLIP preprocessing per batch

**File**: [`src/multimeditron/model/data_loader.py:112–115`](../src/multimeditron/model/data_loader.py)

For the JSONL-based data path (used for NanoVLM/haaissa runs), every batch:
1. Opens a JPEG from disk with PIL (syscall + decompression)
2. `.convert("RGB")` resamples pixel data
3. SigLIP2/CLIP's image processor resizes, normalizes → `float32` tensor

This is parallelized across `dataloader_num_workers: 16`, but it still competes with the GPU pipeline and wastes CPU cycles that could be avoided. The Arrow-based production datasets pre-encode images as byte blobs, avoiding the disk seek, but still do steps 2–3 online.

**Fix options**:
- Pre-process images to tensors offline (as `.npy` or pre-tokenized Arrow datasets) — eliminates all CPU image work at training time. High effort.
- If staying online: ensure `dataloader_pin_memory: true` and `dataloader_prefetch_factor: 4` (already set in production config) to overlap CPU decode with GPU compute.

---

## Part 2 — DDP vs ZeRO-2 for the Full 7-Expert MultiMeditron Pipeline

### 2.1 Model parameter count

| Component | Count | Size (bf16) |
|-----------|-------|-------------|
| Meditron3-8B LLM (LLaMA-3.1-8B, hidden=4096, 32 layers, GQA 32Q/8KV, intermediate=14336, vocab~128K) | 7,505M | 15.0 GB |
| 7 × CLIP-ViT-B/32 experts (hidden=768, 12 layers, patch=32, img=224) | 616M | 1.2 GB |
| 7 × MLP projectors (768→768→4096→4096, 3 linear layers) | 144M | 0.3 GB |
| CrossAttention PEP (dim=4096, heads=8, Q/K/V/O projections) | 67M | 0.1 GB |
| Gating network | ~2M | ~0 GB |
| **Total** | **8,334M ≈ 8.3B** | **16.7 GB** |

> Note: The LLM embedding is counted in the 7.5B figure. Meditron3-8B is reported as "8B" because the medical vocabulary extension enlarges the embedding table beyond vanilla LLaMA-3.1-8B. The figure above (7.5B) uses the base vocabulary (128256 tokens); with an extended vocabulary the true count is slightly higher, consistent with the "8B" label.

---

### 2.2 Training memory formula

For AdamW in mixed precision (bf16 parameters, fp32 optimizer):

| State | Precision | Bytes per param |
|-------|-----------|-----------------|
| Model weights | bf16 | 2 |
| Gradients | bf16 | 2 |
| Optimizer: fp32 param copy | fp32 | 4 |
| Optimizer: 1st moment | fp32 | 4 |
| Optimizer: 2nd moment | fp32 | 4 |
| **Total** | | **16** |

Total memory for all states (before sharding): `8.334B × 16 bytes = 133.3 GB` + activations.

**Activations** (with gradient checkpointing, bs=8, seq_len=2048): ~6 GB estimated.

**Total per GPU (no sharding): ~139 GB** — exceeds the 96 GB GH200 limit.

---

### 2.3 Stage 1 — ALIGNMENT (projectors + CrossAttn only)

In Stage 1, only the 7 projectors (144M), CrossAttention (67M), and gating (2M) are trained. The remaining 8.1B parameters are frozen — their gradients and optimizer states are not stored.

| State | GB |
|-------|----|
| All model weights (bf16, frozen + trainable) | 16.7 |
| Gradients (trainable 213M only) | 0.43 |
| Optimizer (trainable 213M only, fp32×3) | 2.55 |
| Activations (estimated) | ~3 |
| **Total per GPU** | **~23 GB** |

**Conclusion: DDP works for Stage 1 with a comfortable margin on any number of GPUs ≥ 1.**

The current production config (`deepspeed_fast.json`, ZeRO-3) is unnecessary for Stage 1 and incurs collective communication overhead on every parameter access. Switching Stage 1 to no-DeepSpeed DDP would eliminate all ZeRO communication and improve throughput significantly (the smoke tests showed no-DS is 2–5× faster than ZeRO-3 for the equivalent stage).

---

### 2.4 Stage 2 — END2END (all parameters trainable)

#### DDP
Each GPU holds a full copy of all states: **~139 GB per GPU**.  
`139 GB > 96 GB` → **DDP is not viable for Stage 2.** ❌

Even across 4 GPUs (1 node, 384 GB total), DDP still OOMs because each *individual* GPU needs 139 GB, not the node total.

#### ZeRO-2 (shards gradients + optimizer states; each GPU keeps full weights)

`Per GPU = weights(16.7 GB) + (grads + optimizer)(116.6 GB) / N + activations(6 GB)`

| N GPUs | Nodes | Per-GPU memory | Fits (96 GB)? |
|--------|-------|----------------|---------------|
| 2 | 1 | 81.0 GB | ✅ |
| **4** | **1** | **51.8 GB** | **✅ recommended minimum** |
| 8 | 2 | 37.3 GB | ✅ |
| 16 | 4 | 30.0 GB | ✅ |
| 128 | 32 | 23.6 GB | ✅ |
| 512 | 128 | 22.9 GB | ✅ |

ZeRO-2 is viable from **1 node (4 GPUs)** onward. At 1 node each GPU uses ~52 GB, leaving 44 GB of headroom for larger batches or longer sequences.

#### ZeRO-3 (shards weights + gradients + optimizer states — current production config)

`Per GPU = (weights + grads + optimizer)(133.3 GB) / N + activations(6 GB)`

| N GPUs | Nodes | Per-GPU memory | Fits? |
|--------|-------|----------------|-------|
| 2 | 1 | 72.7 GB | ✅ |
| 4 | 1 | 39.3 GB | ✅ |
| 8 | 2 | 22.7 GB | ✅ |
| 512 | 128 | 6.3 GB | ✅ |

ZeRO-3 fits at fewer GPUs than ZeRO-2 but at the cost of an all-gather + reduce-scatter call **per layer per forward/backward pass** — up to 64 collectives per step for a 32-layer LLM. The smoke tests showed ZeRO-3 is ~1.6× slower than ZeRO-2 (1.83 vs 2.94 samples/sec on E2E), and this gap grows with model size.

---

### 2.5 Recommendation

| Training stage | Recommended strategy | Reasoning |
|----------------|----------------------|-----------|
| **Stage 1 (ALIGNMENT)** | **No DeepSpeed (DDP)** | Trainable params are only 213M; total memory ~23 GB/GPU. ZeRO-3 is unnecessary and slows down training with comms overhead. |
| **Stage 2 (END2END)** | **ZeRO-2, ≥ 4 GPUs** | DDP doesn't fit (139 GB > 96 GB). ZeRO-2 fits from 1 node and avoids the per-layer all-gather of ZeRO-3. |

Switching Stage 2 from ZeRO-3 to ZeRO-2 should recover ~1.6× throughput while staying comfortably within memory budget at 1+ nodes. The memory savings from ZeRO-3 over ZeRO-2 are only meaningful when running at very few GPUs (< 4) for this model size.

---

## Summary Table

| Optimization | Effort | Expected Impact |
|---|---|---|
| Move `SamplePreprocessor` out of `torch_call` (fix per-batch deepcopy) | Low | Medium — eliminates CPU stall on every batch |
| Increase batch size from 2→8+ for small-model runs | Config only | High — directly raises GPU occupancy |
| Stage 1: switch from ZeRO-3 to DDP (no DeepSpeed) | Config only | High — eliminates all ZeRO comms on alignment training |
| Stage 2: switch from ZeRO-3 to ZeRO-2 | Config only | Medium (~1.6×) — fewer collective operations per step |
| Pre-compute image tensors offline (Arrow preprocessing) | High | Medium — eliminates online PIL decode + CLIP preprocessing |
