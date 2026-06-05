"""
PathVQA routing analysis — which expert do histopathology images get
routed to by the 5-expert vs 7-expert gating networks?

Loads PathVQA test-split images from HF parquet files and runs them
through both gating models.  Reports top-1 routing breakdown per expert.

Usage (via sbatch inside container — see sbatch_gating_analysis.sh):
    python3 scripts/pathvqa_routing_analysis.py
"""

import random

import torch
from datasets import load_dataset
from PIL import Image

# gating_utils resolves the repo root, sets up sys.path, and exposes the
# (env-overridable) GATING_5EXP / GATING_7EXP path constants.
from gating_utils import GATING_5EXP, GATING_7EXP, load_gating, print_routing, run_gating, short_name

# ── config ────────────────────────────────────────────────────────────────────

N_SAMPLES = 500   # 0 = use all test images
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── image loading via HF datasets ─────────────────────────────────────────────

def load_pathvqa_images(split: str = "test", n: int = 0) -> list:
    """Load PathVQA images via HF datasets (handles broken cache symlinks).

    Args:
        split: "test", "train", or "validation"
        n: max images to load (0 = all)

    Returns:
        list of PIL.Image
    """
    print(f"  Loading split '{split}' via datasets.load_dataset ...")
    ds = load_dataset(
        "flaviagiammarino/path-vqa",
        split=split,
        trust_remote_code=True,
    )
    print(f"  Dataset has {len(ds)} rows")

    images = []
    for row in ds:
        img = row.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            continue
        images.append(img.convert("RGB"))

    # Shuffle deterministically, then trim
    rng = random.Random(SEED)
    rng.shuffle(images)
    if n > 0:
        images = images[:n]

    return images


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("PathVQA Routing Analysis")
    print("Which expert do histopathology images get routed to?")
    print("=" * 72)

    # Load images
    print(f"\nLoading PathVQA test images via HF datasets ...")
    images = load_pathvqa_images(split="test", n=N_SAMPLES)
    print(f"  Loaded {len(images)} images.\n")

    if not images:
        print("[ERROR] No images loaded — aborting.")
        return

    # Load both gating models
    gating_5, names_5 = load_gating(GATING_5EXP, "5-expert", device)
    gating_7, names_7 = load_gating(GATING_7EXP, "7-expert", device)

    # Run analysis
    print(f"\n{'='*72}")
    print("5-EXPERT GATING — PathVQA (histopathology) routing")
    print(f"{'='*72}")
    stats_5 = run_gating(gating_5, names_5, images)
    print_routing(stats_5, names_5, label_width=20)

    print(f"\n{'='*72}")
    print("7-EXPERT GATING — PathVQA (histopathology) routing")
    print(f"{'='*72}")
    stats_7 = run_gating(gating_7, names_7, images)
    print_routing(stats_7, names_7, label_width=20)

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY — Top-1 expert for PathVQA histopathology images")
    print(f"{'='*72}")
    for label, stats, names in [
        ("5-expert", stats_5, names_5),
        ("7-expert", stats_7, names_7),
    ]:
        pcts = stats["routing_pct"]
        top_name = max(names, key=lambda n: pcts[n])
        top_pct = pcts[top_name]
        print(f"  {label}: {short_name(top_name)} ({top_pct:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
