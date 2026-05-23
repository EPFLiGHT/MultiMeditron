"""
extract_debug_batch.py
======================
Run this ONCE to extract a fixed set of 400 samples from the RAW HuggingFace dataset
(HuggingFaceM4/FineVision_concat_shuffled_2) and save them to a shared .pkl file.

Both nanoVLM and MultiMeditron debug modes will load from this same file,
guaranteeing they see bit-for-bit identical RAW inputs.
Each model's own tokenisation/formatting pipeline runs from scratch on these samples,
so any divergence is caught at the exact stage it occurs.

Usage:
    python scripts/extract_debug_batch.py \\
        --out_path /iopsstor/scratch/cscs/haaissa/debug_outputs/debug_batch.pkl \\
        --n_samples 400

Optional args:
    --hf_dataset   HuggingFace dataset repo (default: HuggingFaceM4/FineVision_concat_shuffled_2)
    --hf_split     Dataset split to load (default: train)
    --cache_dir    Local HF cache directory (optional)
"""

import argparse
import io
import os
import pickle


def main(hf_dataset, hf_split, out_path, n_samples, cache_dir):
    from datasets import load_dataset
    from PIL import Image

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading raw dataset '{hf_dataset}' (split='{hf_split}') from HuggingFace...")
    load_kwargs = {"streaming": True}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    ds = load_dataset(hf_dataset, split=hf_split, **load_kwargs)

    # Shuffle with a fixed seed so the 400 samples are always the same
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    samples = []
    for row in ds:
        if len(samples) >= n_samples:
            break

        # FineVision schema: 'images' is a list of PIL Images, 'texts' is a list of
        # {"user": ..., "assistant": ...} dicts.
        images = row.get("images") or []
        texts  = row.get("texts")  or []

        # We only keep samples that have exactly one image (to keep the debug setup simple)
        if len(images) != 1 or len(texts) == 0:
            continue

        img = images[0]
        if img is None:
            continue

        # Convert to PIL if needed, then to raw bytes
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                continue
        if img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Convert FineVision texts format -> standard conversation format
        # [{"user": "...", "assistant": "..."}]  ->  [{"role": "user", "content": "..."},
        #                                              {"role": "assistant", "content": "..."}]
        conversations = []
        for turn in texts:
            user_text = (turn.get("user") or "").strip()
            asst_text = (turn.get("assistant") or "").strip()
            if user_text:
                conversations.append({"role": "user",      "content": user_text})
            if asst_text:
                conversations.append({"role": "assistant", "content": asst_text})

        if not conversations:
            continue

        samples.append({
            "image_bytes":   img_bytes,     # raw PNG bytes (no preprocessing done)
            "conversations": conversations, # clean role/content dicts, NO image tokens
        })

    if len(samples) < n_samples:
        print(f"Warning: Only found {len(samples)} valid single-image samples "
              f"(requested {n_samples}). Consider increasing the streaming buffer.")

    with open(out_path, "wb") as f:
        pickle.dump(samples, f)

    print(f"Saved {len(samples)} samples to {out_path}")
    if samples:
        print(f"Sample 0: {len(samples[0]['image_bytes'])} image bytes, "
              f"{len(samples[0]['conversations'])} conversation turns")
        print(f"  First user turn (preview): {samples[0]['conversations'][0]['content'][:120]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_dataset", default="HuggingFaceM4/FineVision_concat_shuffled_2",
        help="HuggingFace dataset repo ID"
    )
    parser.add_argument(
        "--hf_split", default="train",
        help="Dataset split to load"
    )
    parser.add_argument(
        "--out_path", required=True,
        help="Path to save the .pkl debug batch"
    )
    parser.add_argument(
        "--n_samples", type=int, default=400,
        help="Number of samples to extract"
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="Optional local HF cache directory"
    )
    args = parser.parse_args()
    main(args.hf_dataset, args.hf_split, args.out_path, args.n_samples, args.cache_dir)

