"""
eval_clip_forced_choice.py

Evaluate CLIP-like models using:
1) Text tower diagnostic probe
2) Image tower diagnostic probe
3) 4-way forced-choice image→text retrieval accuracy

It is Top-1 accuracy under fixed negative sampling (chance = 25%).
"""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoProcessor,
)

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


def load_clip_like(model_name_or_path, device):
    """
    Load a CLIP-like model and processor.

    - Vanilla CLIP → CLIPModel + CLIPProcessor
    - BiomedCLIP / custom → AutoModel + AutoProcessor
    """
    lower = model_name_or_path.lower()

    if "biomedclip" in lower:
        model = (
            AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    else:
        model = CLIPModel.from_pretrained(model_name_or_path).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name_or_path)

    # Determine text max length
    if hasattr(model.config, "text_config"):
        max_len = model.config.text_config.max_position_embeddings
    else:
        max_len = getattr(model.config, "max_position_embeddings", 77)

    return model, processor, max_len


# ---------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------


@torch.no_grad()
def get_similarity(text, image_path, model, processor, max_len, device):
    """
    Compute cosine similarity between one image and one text.
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds

    img_emb = torch.nn.functional.normalize(img_emb, dim=1)
    txt_emb = torch.nn.functional.normalize(txt_emb, dim=1)

    return float(torch.matmul(img_emb, txt_emb.T).item())


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    *,
    model_id,
    dataset_path,
    num_samples,
    device,
):
    model, processor, max_len = load_clip_like(model_id, device)

    # -------------------------
    # Text tower probe
    # -------------------------
    probe_texts = [
        "a cat sitting on a couch",
        "retinal fundus image with drusen",
        "an x-ray showing lung consolidation",
        "a photo of a dog running in a field",
        "a patient with diabetic retinopathy",
        "a wooden chair next to a table",
    ]

    dummy_imgs = [Image.new("RGB", (10, 10))] * len(probe_texts)

    probe_inputs = processor(
        text=probe_texts,
        images=dummy_imgs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    probe_out = model(**probe_inputs)
    text_embs = probe_out.text_embeds

    pairwise = torch.nn.functional.cosine_similarity(
        text_embs.unsqueeze(1),
        text_embs.unsqueeze(0),
        dim=-1,
    )

    print("\n=== Text tower probe ===")
    for row in pairwise.cpu().numpy():
        print("  ", " ".join(f"{v:.3f}" for v in row))
    print("unique text embeddings:", len(torch.unique(text_embs.cpu(), dim=0)), "/ 6\n")

    # -------------------------
    # Load dataset
    # -------------------------
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("Dataset is empty")

    N = min(num_samples, len(lines))
    base_dir = dataset_path.parent

    # -------------------------
    # Image tower probe
    # -------------------------
    probe_images = []
    for line in lines[:N]:
        rec = json.loads(line)
        rel = rec["modalities"][0]["value"]
        p = Path(rel)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if p.exists():
            img = Image.open(p)
            if img.mode != "RGB":
                img = img.convert("RGB")
            probe_images.append(img)
        if len(probe_images) >= 6:
            break

    if len(probe_images) >= 2:
        img_inputs = processor(
            text=["dummy"] * len(probe_images),
            images=probe_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        img_out = model(**img_inputs)
        img_embs = img_out.image_embeds

        pairwise_img = torch.nn.functional.cosine_similarity(
            img_embs.unsqueeze(1),
            img_embs.unsqueeze(0),
            dim=-1,
        )

        print("=== Image tower probe ===")
        for row in pairwise_img.cpu().numpy():
            print("  ", " ".join(f"{v:.3f}" for v in row))
        print(
            "unique image embeddings:",
            len(torch.unique(img_embs.cpu(), dim=0)),
            "/",
            len(probe_images),
            "\n",
        )

    # -------------------------
    # 4-way forced-choice retrieval
    # -------------------------
    correct = 0
    used = 0

    for i in range(N):
        candidates = list(range(N))
        candidates.remove(i)
        neg_ids = random.sample(candidates, 3)

        items = [json.loads(lines[i])] + [json.loads(lines[j]) for j in neg_ids]

        texts = [x["text"] for x in items]

        rel = items[0]["modalities"][0]["value"]
        p = Path(rel)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            continue

        sims = [
            get_similarity(t, str(p), model, processor, max_len, device) for t in texts
        ]

        if int(torch.tensor(sims).argmax().item()) == 0:
            correct += 1
        used += 1

    if used == 0:
        raise RuntimeError("No valid samples evaluated")

    return correct / used


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = [
        ("BASE CLIP", "openai/clip-vit-base-patch32"),
        ("BiomedCLIP", "chuhac/BiomedCLIP-vit-bert-hf"),
    ]

    for name, model_id in models:
        print(f"\n===== Evaluating {name} ({model_id}) =====")
        acc = evaluate_model(
            model_id=model_id,
            dataset_path=args.dataset,
            num_samples=args.num_samples,
            device=args.device,
        )
        print(f"{name} accuracy: {acc:.4f}\n")


if __name__ == "__main__":
    main()
