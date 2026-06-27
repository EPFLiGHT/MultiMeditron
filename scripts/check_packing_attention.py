"""
Layer 3 — Attention isolation check for packed sequences.

Verifies that after sequence packing, attention scores across the sub-sequence
boundary are negligible (< threshold). Without true varlen-FA2 attention masking
the model uses causal attention, which *will* leak across boundaries. This script
quantifies the leakage so you know exactly what to fix before enabling varlen.

Usage (on any node with the training container):
    cd /users/surech/meditron/MultiMeditron
    PYTHONPATH=src python scripts/check_packing_attention.py \\
        --checkpoint /iopsstor/scratch/cscs/surech/multimeditron/checkpoints/unfreeze/attn_pep/\\
MultiMeditron-8B-attn-pep-end2end-7exp/checkpoint-800 \\
        --max_length 64 \\
        --threshold 1e-3

The script loads just the language model backbone (no vision tower) so it can run
on a single GPU or even CPU (slower).

Exit code: 0 if max cross-boundary attention ≤ threshold, 1 otherwise.
"""

import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")


def build_fake_packed_batch(seq_len_a: int, seq_len_b: int, max_length: int, vocab_size: int = 32000):
    """
    Build a minimal packed batch containing two fake sub-sequences A and B.

    Returns:
        input_ids:      (1, max_length)
        attention_mask: (1, max_length)  — 1 for real tokens, 0 for padding
        position_ids:   (1, max_length)  — restarted at 0 for each sub-sequence
        cu_seqlens:     list of one 1-D int32 tensor [0, len_a, len_a+len_b, max_length]
    """
    assert seq_len_a + seq_len_b <= max_length, "Sequences together exceed max_length"

    ids_a = torch.randint(1, vocab_size, (seq_len_a,))
    ids_b = torch.randint(1, vocab_size, (seq_len_b,))
    pad_len = max_length - seq_len_a - seq_len_b

    input_ids = torch.cat([ids_a, ids_b, torch.zeros(pad_len, dtype=torch.long)]).unsqueeze(0)
    attn_mask = torch.cat([
        torch.ones(seq_len_a + seq_len_b, dtype=torch.long),
        torch.zeros(pad_len, dtype=torch.long)
    ]).unsqueeze(0)

    pos_ids = torch.cat([
        torch.arange(seq_len_a),
        torch.arange(seq_len_b),
        torch.zeros(pad_len, dtype=torch.long)
    ]).unsqueeze(0)

    cu_seqlens = torch.tensor([0, seq_len_a, seq_len_a + seq_len_b, max_length], dtype=torch.int32)

    return input_ids, attn_mask, pos_ids, cu_seqlens


def check_attention_leakage(checkpoint: str, max_length: int, threshold: float, device: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"Loading config from {checkpoint} ...")
    # Load LLM backbone only (skip vision tower to keep memory low)
    try:
        from multimeditron.model.model import MultiModalModelForCausalLM
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        model = MultiModalModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",  # eager so output_attentions=True works
        )
    except Exception as e:
        print(f"Failed to load MultiModalModelForCausalLM: {e}")
        print("Falling back to raw AutoModelForCausalLM (text-only, no vision tower)")
        config = AutoConfig.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )

    model.eval()

    seq_a = max_length // 3
    seq_b = max_length // 3
    input_ids, attn_mask, pos_ids, cu_seqlens = build_fake_packed_batch(seq_a, seq_b, max_length)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    pos_ids = pos_ids.to(device)

    print(f"Running forward pass with output_attentions=True ...")
    print(f"  Sub-seq A: tokens 0..{seq_a-1}")
    print(f"  Sub-seq B: tokens {seq_a}..{seq_a+seq_b-1}")
    print(f"  Padding:   tokens {seq_a+seq_b}..{max_length-1}")

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            output_attentions=True,
        )

    # out.attentions: tuple of (batch=1, heads, seq, seq) per layer
    # Check last layer (most semantic) and first layer (most structural)
    layers_to_check = [0, len(out.attentions) - 1]
    boundary = seq_a

    results = []
    for layer_idx in layers_to_check:
        attn = out.attentions[layer_idx]  # (1, heads, seq, seq)
        # Cross-boundary: B attending to A
        # attn[0, :, boundary:boundary+seq_b, :boundary]  → shape (heads, seq_b, seq_a)
        cross_ba = attn[0, :, boundary:boundary + seq_b, :boundary]
        max_cross = cross_ba.abs().max().item()
        mean_cross = cross_ba.abs().mean().item()

        # Within-boundary (control): A attending to A
        within_aa = attn[0, :, :seq_a, :seq_a]
        max_within = within_aa.abs().max().item()

        results.append({
            "layer": layer_idx,
            "max_cross_BA": max_cross,
            "mean_cross_BA": mean_cross,
            "max_within_AA": max_within,
            "ratio_cross_to_within": max_cross / (max_within + 1e-10),
        })

        print(f"\n  Layer {layer_idx}:")
        print(f"    max cross-boundary (B→A):  {max_cross:.2e}")
        print(f"    mean cross-boundary (B→A): {mean_cross:.2e}")
        print(f"    max within A (control):    {max_within:.2e}")
        print(f"    cross/within ratio:         {max_cross / (max_within + 1e-10):.4f}")

    max_cross_all = max(r["max_cross_BA"] for r in results)

    print(f"\n{'='*60}")
    if max_cross_all <= threshold:
        print(f"PASS: max cross-boundary attention {max_cross_all:.2e} ≤ threshold {threshold:.2e}")
        print("Attention isolation is adequate.")
        return True
    else:
        print(f"WARN: max cross-boundary attention {max_cross_all:.2e} > threshold {threshold:.2e}")
        print("Cross-sample attention leakage detected.")
        print("This is expected with standard causal attention (no varlen masking).")
        print("To fix: pass cu_seqlens to flash_attn_varlen_func in model.py.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check attention isolation in packed sequences.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a MultiMeditron or LLaMA checkpoint directory.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Sequence length to test (smaller = faster, default 64).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Max acceptable cross-boundary attention score (default 1e-3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    args = parser.parse_args()

    print(f"Attention isolation check")
    print(f"  checkpoint:  {args.checkpoint}")
    print(f"  max_length:  {args.max_length}")
    print(f"  threshold:   {args.threshold}")
    print(f"  device:      {args.device}")
    print()

    ok = check_attention_leakage(
        checkpoint=args.checkpoint,
        max_length=args.max_length,
        threshold=args.threshold,
        device=args.device,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
