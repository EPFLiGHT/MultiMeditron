"""
This script performs qualitative image-to-image retrieval analysis using the vision tower of a CLIP-style vision–text 
dual encoder. Given a JSONL evaluation dataset, it computes normalized image embeddings for all samples, selects a query 
image (optionally constrained to specific labels) and retrieves the top-K most similar images based on cosine similarity 
in embedding space. The results are visualized as a single figure showing the query alongside its nearest neighbors, 
enabling manual inspection of semantic clustering, failure modes, and expert specialization behavior. 

This tool is intended for qualitative evaluation and sanity checking of learned visual representations rather than 
quantitative benchmarking.
"""

import json
import random
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from transformers import VisionTextDualEncoderProcessor

from load_from_clip import load_model  # your helper

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset_images(jsonl_path, max_samples=None):
    """
    Load images + their labels from a JSONL file.

    Each line like:
      {
        "text": "<label>",
        "modalities": [{"type": "image", "value": "images/xxx.jpg"}]
      }
    """
    images = []
    labels = []

    base_dir = Path(jsonl_path).parent
    jsonl_path = Path(jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed JSON at line {line_no}")
                continue

            modalities = sample.get("modalities", [])
            if not modalities:
                continue

            rel_path = modalities[0].get("value")
            if not rel_path:
                continue

            p = Path(rel_path)
            if not p.is_absolute():
                p = (base_dir / p).resolve()

            if not p.exists():
                continue

            try:
                img = Image.open(p)
                if img.mode != "RGB":
                    img = img.convert("RGB")
            except Exception as e:
                print(f"[WARN] Could not open image {p}: {e}")
                continue

            label = sample.get("text", "unknown")

            images.append(img)
            labels.append(label)

            if max_samples is not None and len(images) >= max_samples:
                break

    print(f"[INFO] Loaded {len(images)} images from {jsonl_path}")
    return images, labels


@torch.no_grad()
def compute_image_embeds(model, processor, images, batch_size=32):
    """
    Compute normalized image embeddings using the vision tower only.
    """
    all_embs = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]

        inputs = processor(
            images=batch,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)

        image_embs = model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs.image_embeds if hasattr(image_embs, "image_embeds") else image_embs.pooler_output
        embs = torch.nn.functional.normalize(image_embs, dim=-1)

        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)  # (N, D)


def pick_query_index(labels, preferred_query_labels):
    """
    Pick an index whose label is in preferred_query_labels.
    If none exists, fall back to any random index.
    """
    candidate_indices = [
        i for i, lab in enumerate(labels) if lab in preferred_query_labels
    ]

    if candidate_indices:
        random.seed(0)
        q_idx = random.choice(candidate_indices)
        print(f"[INFO] Picked query from preferred labels: {labels[q_idx]}")
        return q_idx
    else:
        print("[WARN] No labels from preferred set found, using random query.")
        random.seed(0)
        return random.randrange(len(labels))


def visualize_retrieval(
    model_name_or_path,
    eval_dataset,
    k=3,
    preferred_query_labels=None,
    out_path=None,
):
    # 1) load model & processor
    print(f"[INFO] Loading model from: {model_name_or_path}")
    model = load_model(model_name_or_path).to(device).eval()
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_name_or_path)

    # 2) load ALL images + labels
    images, labels = load_dataset_images(
        eval_dataset,
        max_samples=None,  # or cap if too big
    )

    if len(images) < k + 1:
        raise ValueError(
            f"Not enough images ({len(images)}) to do retrieval with k={k}."
        )

    # 3) compute embeddings for the whole dataset
    embs = compute_image_embeds(model, processor, images)  # (N, D)

    # 4) pick a query index
    if preferred_query_labels is not None:
        query_idx = pick_query_index(labels, preferred_query_labels)
    else:
        random.seed(0)
        query_idx = random.randrange(len(images))

    query_emb = embs[query_idx : query_idx + 1]  # (1, D)

    # 5) cosine similarity query vs all
    sims = torch.matmul(query_emb, embs.T).squeeze(0)  # (N,)
    ranked = torch.argsort(sims, descending=True).tolist()

    # remove self from ranking
    ranked = [i for i in ranked if i != query_idx]
    topk = ranked[:k]

    # 6) plot query + top-k neighbors
    total = k + 1
    plt.figure(figsize=(3 * total, 4))

    # query
    plt.subplot(1, total, 1)
    plt.imshow(images[query_idx])
    plt.axis("off")
    plt.title(f"Query\n{labels[query_idx]}", fontsize=9)

    # retrieved images
    for j, idx in enumerate(topk, start=2):
        plt.subplot(1, total, j)
        plt.imshow(images[idx])
        plt.axis("off")
        sim_value = sims[idx].item()

        # take only the first 3 words of the label
        full_label = labels[idx]
        short_label = " ".join(str(full_label).split()[:3])

        title = f"Top {j-1}\n{short_label}\ncos={sim_value:.2f}"
        plt.title(title, fontsize=9)
        plt.tight_layout()

    if out_path is None:
        out_path = Path("most_sim.png")
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved retrieval figure to: {out_path}")


if __name__ == "__main__":
    # Best ophthalmology expert model (from earlier results)
    OPHTH_MODEL = (
        "/lightscratch/users/turan/training/models_opthalmology/"
        "combined_dataset_opthalmology_regularization_focused_config_1"
    )

    # Eye datasets – adjust paths if needed
    configs = [
        (
            "messidor",
            "/lightscratch/users/turan/datasets/messidor2_eval/messidor_val_raw.jsonl",
        )
    ]

    for slug, ds_path in configs:
        visualize_retrieval(
            model_name_or_path=OPHTH_MODEL,
            eval_dataset=ds_path,
            k=3,
            preferred_query_labels={"Moderate diabetic retinopathy without macular edema"},
            out_path=f"/lightscratch/users/turan/evaluation_clip/most_sim_ophthal_{slug}.png",
        )
