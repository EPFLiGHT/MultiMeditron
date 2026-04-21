"""
PathVQA routing analysis — which expert do histopathology images get
routed to by the 5-expert vs 7-expert gating networks?

Loads PathVQA test-split images from HF parquet files and runs them
through both gating models.  Reports top-1 routing breakdown per expert.

Usage (via sbatch inside container — see sbatch_gating_analysis.sh):
    python3 scripts/pathvqa_routing_analysis.py
"""

import sys
import os
import random
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

# ── config ────────────────────────────────────────────────────────────────────

GATING_5EXP = (
    "/capstor/store/cscs/swissai/a127/meditron/hf_cache/hub/"
    "models--ClosedMeditron--MultiMeditron-Gating/snapshots/"
    "e1d1310b6e1962857b61b0009b9a4d7e196e84fa"
)
GATING_7EXP = (
    "/users/surech/meditron/MultiMeditron/models/CLIP/MultiMeditron-Gating"
)

N_SAMPLES = 500   # 0 = use all test images
SEED = 42

EXPERT_LABELS = {
    "MedExpert-CT": "CT",
    "clip-vit-base-patch32": "Generalist",
    "MedExpert-MRI": "MRI",
    "MedExpert-Ultrasound": "Ultrasound",
    "checkpoint-4350": "Ultrasound",
    "MedExpert-Xray": "X-ray",
    "OphthalmologyExpert": "Ophthalmology",
    "SkinExpert": "Skin",
}


def short_name(full_path: str) -> str:
    basename = os.path.basename(full_path.rstrip("/"))
    return EXPERT_LABELS.get(basename, basename)


# ── paths ─────────────────────────────────────────────────────────────────────

sys.path.insert(0, "/users/surech/meditron/MultiMeditron/src")
from multimeditron.model.modalities.moe.gating import GatingNetwork

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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


# ── gating inference ──────────────────────────────────────────────────────────

def run_gating(gating_model, class_names: list, images: list) -> dict:
    vote_counter = Counter()
    score_accum = defaultdict(float)
    batch_size = 32

    for start in range(0, len(images), batch_size):
        batch = images[start: start + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)

        with torch.no_grad():
            logits, topk_indices, weights = gating_model(tensors)

        top1 = topk_indices[:, 0].cpu().tolist()
        for idx in top1:
            vote_counter[class_names[idx]] += 1

        w = weights.cpu()
        for c_idx, name in enumerate(class_names):
            score_accum[name] += w[:, c_idx].sum().item()

    n = len(images)
    return {
        "n": n,
        "routing_pct": {name: 100.0 * vote_counter[name] / n for name in class_names},
        "avg_weights": {name: score_accum[name] / n for name in class_names},
    }


# ── print helpers ─────────────────────────────────────────────────────────────

def print_routing(stats: dict, class_names: list):
    pcts = stats["routing_pct"]
    wgts = stats["avg_weights"]
    print(f"  {'Expert':<20} {'Top-1 %':>8}  {'Avg weight':>10}  Bar")
    print(f"  {'-'*60}")
    for name in sorted(class_names, key=lambda n: -pcts[n]):
        label = short_name(name)
        pct = pcts[name]
        wgt = wgts[name]
        bar = "█" * int(pct / 2)
        print(f"  {label:<20} {pct:>7.1f}%  {wgt:>10.4f}  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def load_gating(path: str, label: str):
    print(f"\nLoading {label} gating from:\n  {path}")
    m = GatingNetwork.from_pretrained(path)
    m.eval().to(device)
    names = m.config.class_names
    n = m.config.num_classes
    short = [short_name(x) for x in names]
    print(f"  {n} experts: {short}")
    return m, names


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
    gating_5, names_5 = load_gating(GATING_5EXP, "5-expert")
    gating_7, names_7 = load_gating(GATING_7EXP, "7-expert")

    # Run analysis
    print(f"\n{'='*72}")
    print("5-EXPERT GATING — PathVQA (histopathology) routing")
    print(f"{'='*72}")
    stats_5 = run_gating(gating_5, names_5, images)
    print_routing(stats_5, names_5)

    print(f"\n{'='*72}")
    print("7-EXPERT GATING — PathVQA (histopathology) routing")
    print(f"{'='*72}")
    stats_7 = run_gating(gating_7, names_7, images)
    print_routing(stats_7, names_7)

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
