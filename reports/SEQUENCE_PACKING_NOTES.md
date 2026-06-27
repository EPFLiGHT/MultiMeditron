# Sequence Packing — meeting notes (for discussion with Fabrice)

> Prep for the sequence-packing discussion. Covers the configuration used, the
> open training-collapse issue, and the Pixel Shuffle projector question.

## 1. What sequence packing does here

Standard training pads every batch to the longest sequence in it; with
variable-length medical VQA data, >60% of a batch can be padding. Packing bins
multiple short samples into one fixed-length sequence (`max_sequence_length`) and
tells Flash-Attention-2 the sub-sequence boundaries so samples don't attend
across each other.

- **Collator** (`src/multimeditron/model/data_loader.py`, `_pack_sequences`):
  first-fit bin packing; each bin padded to `max_sequence_length`; emits a
  `cu_seqlens` tensor `[0, s₀, s₀+s₁, …, Σsᵢ, max_len]` per bin.
- **FA2 fix** (`src/multimeditron/train/trainer.py`): HF derives `cu_seqlens`
  from `attention_mask.sum(-1)`, which treats a packed bin as one sequence →
  **cross-sample attention leakage**. We monkey-patch `_get_unpad_data` to use the
  collator's `cu_seqlens` (thread-local `_PACKING_CONTEXT`), so FA2 respects
  individual sample boundaries. Patch location is printed at startup
  (`[PACKING PATCH] locations=…`). Also forces `cu_seqlens` to int32
  (`flash_attn_varlen_func` requirement).

## 2. Configuration used

`cookbook/sft/moe/attn/pep/stage2_sanitycheck_zero2_packed.yaml`:

| Setting | Value |
|---|---|
| `pack_sequences` | `true` |
| `max_sequence_length` | `4096` |
| DeepSpeed / ZeRO | **ZeRO-2** (`config/deepspeed_zero2.json`) |
| Nodes / GPUs | 4 nodes / 16 GPUs |
| `per_device_train_batch_size` | 8 |
| `gradient_accumulation_steps` | 2 |
| Model | `moe_meditron_clip_pep` (7-expert, **per-expert MLP projectors**) |
| Datasets | 11 (`BUSI, COVID_US, ct2, iu_xray, PMC_VQA_FULL, llava_instruct, medtrinity×2, image_mammoth, eye_dataset_converted, skin_dataset_converted`) |

It is identical to `stage2_sanitycheck_zero2.yaml` except `pack_sequences: true`, so
the two give a clean packed-vs-unpacked comparison (4 nodes, ~10 min each).

## 3. Open issue — training collapse (job 2340502)

The packed ZeRO-2 run collapsed:
- Steps 1–2: `loss ≈ 15.5`, `grad_norm ≈ 310` (normal warm-up).
- Steps 3–200: `loss = 0.0` exactly, `grad_norm = 1.4142` (= √2).

The √2 grad-norm is suspicious (looks like only a weight-decay term remains).
**Status: root cause still open.** The original hypothesis (all labels masked to
−100 by the packing collator) does **not** hold up on inspection — in
`_pack_sequences` only the first token of each non-first sub-sequence and the
padding region are set to `IGNORE_TOKEN_INDEX`; real labels survive. So the
collapse is more likely in the FA2 patch ↔ ZeRO-2 interaction or `position_ids`
handling, not label masking. **Question for Fabrice:** has he seen this signature
with FA2 varlen under ZeRO-2?

The unpacked baseline (job 2340681) was launched in parallel for the loss-curve
comparison; packed-vs-unpacked MFU is blocked until the collapse is fixed.

## 4. Pixel Shuffle projector — was it used?

**No — not in any packing run.**

- Pixel Shuffle is implemented (`model/projectors/pixel_shuffle.py`) and wired
  into the **single-CLIP** `ImageModality` (`projection_type: pixel_shuffle`,
  reduces token count by `factor²`).
- It was only exercised in separate **single-encoder smoketests**
  (`cookbook/sft/single_clip/qwen_biomedclip/…`: SigLIP2-512 + SmolLM2,
  `pixel_shuffle_factor: 4`).
- The packing config uses `moe_meditron_clip_pep`, i.e. the **per-expert MLP
  projectors** — Pixel Shuffle is not on that path. Packing and Pixel Shuffle
  were never combined.

## 5. Discussion points

1. Root-causing the loss=0 collapse (FA2 varlen × ZeRO-2 × position_ids).
2. Whether to pursue Pixel Shuffle in the MoE PEP path (fewer image tokens →
   shorter sequences → better packing density).
3. Target `max_sequence_length` and expected padding-fraction reduction.
4. Moving production Stage 2 from ZeRO-3 to ZeRO-2 (the 1.5% MFU finding).
