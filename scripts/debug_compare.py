"""
debug_compare.py
================
Loads the .npy files saved by debug_nanovlm.py and debug_multimeditron.py
and prints a side-by-side comparison at each pipeline stage.

Usage:
    python scripts/debug_compare.py \
        --out_dir /iopsstor/scratch/cscs/haaissa/debug_outputs
"""

import argparse, os
import numpy as np


STAGES = [
    ("tokenizer_input_ids", "Tokenizer → input_ids (token IDs)"),
    ("tokenizer_labels",    "Tokenizer → labels (loss mask)"),
    ("encoder_features",    "SigLIP encoder output"),
    ("projector_output",    "Pixel Shuffle projector output"),
    ("lm_logits",           "LM head logits"),
    ("loss",                "Training loss (scalar per sample)"),
]


def load(path):
    arr = np.load(path, allow_pickle=True)
    # If it's an array of arrays (variable length), flatten first elem for stats
    if arr.dtype == object:
        return arr
    return arr.astype(np.float32)


def stat_str(arr):
    """Return a compact stats string for an array or array-of-arrays."""
    if arr.dtype == object:
        # Take stats over first 10 samples
        flat = np.concatenate([a.flatten().astype(np.float32) for a in arr[:10]])
    else:
        flat = arr.flatten().astype(np.float32)
    return (f"mean={flat.mean():.5f}  std={flat.std():.5f}  "
            f"min={flat.min():.5f}  max={flat.max():.5f}  "
            f"first5={flat[:5].tolist()}")


def shape_str(arr):
    if arr.dtype == object:
        return f"[{len(arr)} x {list(arr[0].shape)}]"
    return str(list(arr.shape))


def main(out_dir):
    nano_dir  = os.path.join(out_dir, "nanovlm")
    multi_dir = os.path.join(out_dir, "multimeditron")

    SEP = "─" * 80
    print(SEP)
    print("  PIPELINE COMPARISON: nanoVLM  vs  MultiMeditron")
    print(f"  nanoVLM      dir: {nano_dir}")
    print(f"  MultiMeditron dir: {multi_dir}")
    print(SEP)

    first_divergence = None

    for fname, label in STAGES:
        n_path = os.path.join(nano_dir,  fname + ".npy")
        m_path = os.path.join(multi_dir, fname + ".npy")

        if not os.path.exists(n_path):
            print(f"\n⚠️  MISSING nanoVLM file: {fname}.npy")
            continue
        if not os.path.exists(m_path):
            print(f"\n⚠️  MISSING MultiMeditron file: {fname}.npy")
            continue

        n = load(n_path)
        m = load(m_path)

        print(f"\n{'═'*60}")
        print(f"  STAGE: {label}")
        print(f"{'─'*60}")
        print(f"  nanoVLM       shape={shape_str(n)}")
        print(f"                {stat_str(n)}")
        print(f"  MultiMeditron shape={shape_str(m)}")
        print(f"                {stat_str(m)}")

        # Compare
        # NanoVLM shifts its labels left by 1 manually before saving.
        # MultiMeditron saves unshifted labels (HuggingFace models shift internally).
        # We align MultiMeditron labels with NanoVLM for a fair comparison:
        if fname == "tokenizer_labels" and m.dtype == object:
            for i in range(len(m)):
                m[i] = np.roll(m[i], -1)
                m[i][-1] = -100
        elif fname == "tokenizer_labels" and m.dtype != object:
            m = np.roll(m, -1, axis=-1)
            m[..., -1] = -100

        n_flat = np.concatenate([a.flatten().astype(np.float32) for a in n[:10]]) \
                 if n.dtype == object else n.flatten().astype(np.float32)
        m_flat = np.concatenate([a.flatten().astype(np.float32) for a in m[:10]]) \
                 if m.dtype == object else m.flatten().astype(np.float32)

        # Shape mismatch
        if n.dtype == object and m.dtype == object:
            n_s = list(n[0].shape)
            m_s = list(m[0].shape)
            if n_s != m_s:
                print(f"\n  ❌ SHAPE MISMATCH: nanoVLM {n_s}  vs  MultiMeditron {m_s}")
                if first_divergence is None:
                    first_divergence = label
                continue
            
            # Print exact first mismatch for token IDs
            if fname == "tokenizer_input_ids":
                for i in range(len(n_flat)):
                    if n_flat[i] != m_flat[i]:
                        print(f"  ❌ FIRST MISMATCH at index {i}: nanoVLM={n_flat[i]}, MultiMeditron={m_flat[i]}")
                        break

        # Mean/std difference
        mean_diff = abs(n_flat.mean() - m_flat.mean())
        std_diff  = abs(n_flat.std()  - m_flat.std())
        max_diff  = abs(n_flat - m_flat[:len(n_flat)]).max() if len(n_flat) == len(m_flat) else float('nan')

        if mean_diff < 0.001 and std_diff < 0.05:
            print(f"  ✅ MATCH  |mean_diff|={mean_diff:.5f}  |std_diff|={std_diff:.5f}")
        elif mean_diff < 0.01 and std_diff < 0.2:
            print(f"  ⚠️  CLOSE  |mean_diff|={mean_diff:.5f}  |std_diff|={std_diff:.5f}")
        else:
            print(f"  ❌ DIVERGES  |mean_diff|={mean_diff:.5f}  |std_diff|={std_diff:.5f}  max_elem_diff={max_diff:.5f}")
            if first_divergence is None:
                first_divergence = label

    print(f"\n{'═'*60}")
    if first_divergence:
        print(f"  🔍 FIRST DIVERGENCE AT: {first_divergence}")
        print(f"  ↳ Everything before this stage is equivalent.")
        print(f"  ↳ The bug is introduced at or before this stage.")
    else:
        print(f"  ✅ All stages match — pipelines are equivalent.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.out_dir)

