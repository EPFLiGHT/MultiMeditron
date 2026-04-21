"""
Train a 7-class GatingNetwork (ResNet50-based) that routes medical images
to the correct expert CLIP encoder.

Fixes over the old image_router_train.py:
  - Uses GatingNetwork (PreTrainedModel) directly — save_pretrained() emits
    resnet.*-prefixed state_dict + config.json with class_names automatically
  - Loads Arrow datasets via pyarrow IPC (no HF datasets dependency)
  - Supports both 'image' and 'modalities' column formats
  - DDP multi-GPU via torchrun
  - WandB logging
  - Proper train/val split, per-class accuracy, cosine LR with warmup

Single GPU:
    python scripts/train_gating.py --config scripts/config_gating.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 scripts/train_gating.py --config scripts/config_gating.yaml

Override any config value from CLI:
    torchrun --nproc_per_node=4 scripts/train_gating.py \\
        --config scripts/config_gating.yaml \\
        --num_epochs 30 --lr 3e-4 --batch_size 64
"""

import argparse
import io
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import pyarrow.ipc as ipc
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import yaml

# ── sys.path setup ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third-party" / "lmms-eval"))

from multimeditron.model.modalities.moe.gating import GatingNetwork, GatingNetworkConfig


# ─── Default class mapping (7 classes) ───────────────────────────────────────

DEFAULT_CLASS_NAMES = [
    "/users/surech/meditron/MultiMeditron/models/CLIP/ClosedMeditron/MedExpert-CT",
    "/capstor/store/cscs/swissai/a127/meditron/models/openai/clip-vit-base-patch32",
    "/users/surech/meditron/MultiMeditron/models/CLIP/ClosedMeditron/MedExpert-MRI",
    "/capstor/store/cscs/swissai/a127/meditron/models/CLIP/UltraSoundCLIP/checkpoint-4350",
    "/users/surech/meditron/MultiMeditron/models/CLIP/ClosedMeditron/MedExpert-Xray",
    "/capstor/store/cscs/swissai/a127/meditron/models/CLIP/OphthalmologyExpert",
    "/capstor/store/cscs/swissai/a127/meditron/models/CLIP/SkinExpert",
]

DEFAULT_CLASS_LABELS = ["CT", "General", "MRI", "Ultrasound", "Xray", "Ophthalmology", "Skin"]


# ─── Arrow image loading ─────────────────────────────────────────────────────

def _read_arrow_table(arrow_file: str):
    """Read an Arrow file, trying IPC stream first, then file format."""
    try:
        with open(arrow_file, "rb") as f:
            return ipc.open_stream(f).read_all()
    except Exception:
        with open(arrow_file, "rb") as f:
            return ipc.open_file(f).read_all()


def _arrow_dir(dataset_path: str) -> str:
    """Return the directory containing .arrow files (check for train/ subdir)."""
    train_dir = os.path.join(dataset_path, "train")
    return train_dir if os.path.isdir(train_dir) else dataset_path


def _list_arrow_files(directory: str) -> List[str]:
    """List sorted .arrow data files in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("data-") and f.endswith(".arrow")
    )


def _detect_column_format(dataset_path: str) -> Optional[str]:
    """Peek at the first arrow file to detect 'image' vs 'modalities' column."""
    adir = _arrow_dir(dataset_path)
    files = _list_arrow_files(adir)
    if not files:
        return None
    try:
        table = _read_arrow_table(files[0])
        if "image" in table.schema.names:
            return "image"
        if "modalities" in table.schema.names:
            return "modalities"
    except Exception:
        pass
    return None


def load_images_from_arrow(dataset_path: str, max_samples: int) -> List[PIL.Image.Image]:
    """Load up to max_samples PIL images from an Arrow dataset on disk."""
    col_format = _detect_column_format(dataset_path)
    if col_format is None:
        return []

    adir = _arrow_dir(dataset_path)
    arrow_files = _list_arrow_files(adir)
    images: List[PIL.Image.Image] = []

    for arrow_file in arrow_files:
        if len(images) >= max_samples:
            break
        try:
            table = _read_arrow_table(arrow_file)
        except Exception:
            continue

        if col_format == "image":
            col = table.column("image")
            for i in range(len(col)):
                if len(images) >= max_samples:
                    break
                try:
                    row = col[i].as_py()
                    img_bytes = row.get("bytes") if isinstance(row, dict) else None
                    if img_bytes is None:
                        continue
                    images.append(PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                except Exception:
                    continue

        elif col_format == "modalities":
            col = table.column("modalities")
            for i in range(len(col)):
                if len(images) >= max_samples:
                    break
                try:
                    modalities = col[i].as_py()
                    for mod in modalities:
                        if mod.get("type") == "image":
                            img_bytes = mod["value"]["bytes"]
                            images.append(
                                PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            )
                            break  # one image per row
                except Exception:
                    continue

    return images


# ─── Dataset ─────────────────────────────────────────────────────────────────

class GatingImageDataset(Dataset):
    """
    In-memory dataset of (PIL_image, class_label) pairs.
    Applies a torchvision transform at __getitem__ time.
    """

    def __init__(self, images: List[PIL.Image.Image], labels: List[int],
                 transform: transforms.Compose):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.transform(self.images[idx]), self.labels[idx]


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── DDP helpers ─────────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize DDP if launched via torchrun. Returns (rank, local_rank, world_size, is_dist)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str):
    if is_main(rank):
        print(msg, flush=True)


# ─── Data building ───────────────────────────────────────────────────────────

def build_datasets(
    cfg: dict,
    max_samples_override: Optional[int],
    val_split: float,
    test_split: float,
    seed: int,
    rank: int,
) -> Tuple[GatingImageDataset, GatingImageDataset, GatingImageDataset, int, List[str]]:
    """
    Load images from dataset_class_map, split into train/val/test, return datasets.

    The test split is carved off first (before train/val) so the held-out test
    set is deterministic and never influences training. Pass test_split=0 to
    disable the held-out test set (returns an empty dataset as the third element).
    """
    dataset_class_map: Dict = cfg.get("dataset_class_map", {})
    class_labels: List[str] = cfg.get("class_labels", DEFAULT_CLASS_LABELS)
    num_classes: int = cfg.get("num_classes", len(dataset_class_map))
    max_per_class = max_samples_override if max_samples_override is not None else cfg.get("max_samples_per_class", 0)

    all_images: List[PIL.Image.Image] = []
    all_labels: List[int] = []
    rng = random.Random(seed)

    for class_idx_raw, paths in sorted(dataset_class_map.items(), key=lambda x: int(x[0])):
        class_idx = int(class_idx_raw)
        class_label = class_labels[class_idx] if class_idx < len(class_labels) else str(class_idx)
        class_images: List[PIL.Image.Image] = []

        for dpath in paths:
            dpath = dpath.strip()
            if not os.path.exists(dpath):
                log(rank, f"  [WARN] Path does not exist, skipping: {dpath}")
                continue
            remaining = (max_per_class - len(class_images)) if max_per_class > 0 else 999_999
            if remaining <= 0:
                break
            log(rank, f"  Loading class {class_idx} ({class_label}) from {dpath} ...")
            imgs = load_images_from_arrow(dpath, remaining)
            class_images.extend(imgs)
            log(rank, f"    → {len(imgs)} images (running total: {len(class_images)})")

        # Cap per class
        if max_per_class > 0 and len(class_images) > max_per_class:
            rng.shuffle(class_images)
            class_images = class_images[:max_per_class]

        log(rank, f"  Class {class_idx} ({class_label}): {len(class_images)} images total")
        all_images.extend(class_images)
        all_labels.extend([class_idx] * len(class_images))

    # Shuffle and split — test is carved off first so it is always the same
    combined = list(zip(all_images, all_labels))
    rng.shuffle(combined)

    n_test = int(len(combined) * test_split) if test_split > 0 else 0
    test_pairs  = combined[len(combined) - n_test :] if n_test > 0 else []
    remaining   = combined[: len(combined) - n_test]

    n_val = int(len(remaining) * val_split)
    train_pairs = remaining[: len(remaining) - n_val]
    val_pairs   = remaining[len(remaining) - n_val :]

    train_imgs, train_lbls = (list(t) for t in zip(*train_pairs)) if train_pairs else ([], [])
    val_imgs,   val_lbls   = (list(t) for t in zip(*val_pairs))   if val_pairs   else ([], [])
    test_imgs,  test_lbls  = (list(t) for t in zip(*test_pairs))  if test_pairs  else ([], [])

    train_ds = GatingImageDataset(train_imgs, train_lbls, get_train_transform())
    val_ds   = GatingImageDataset(val_imgs,   val_lbls,   get_val_transform())
    test_ds  = GatingImageDataset(test_imgs,  test_lbls,  get_val_transform())

    log(rank, f"\n  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")

    # Print class distributions
    if is_main(rank):
        splits = [("Train", train_lbls), ("Val", val_lbls)]
        if test_lbls:
            splits.append(("Test (held-out)", test_lbls))
        for split_name, labels in splits:
            counts = Counter(labels)
            print(f"  {split_name} class distribution:")
            for c in range(num_classes):
                lbl = class_labels[c] if c < len(class_labels) else str(c)
                print(f"    {lbl}: {counts.get(c, 0)}")

    return train_ds, val_ds, test_ds, num_classes, class_labels


# ─── Training ────────────────────────────────────────────────────────────────

def train(config: dict):
    rank, local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    log(rank, f"\n{'='*60}")
    log(rank, f"  GatingNetwork Training")
    log(rank, f"  Distributed: {is_distributed}  World size: {world_size}")
    log(rank, f"  Device: {device}")
    log(rank, f"{'='*60}")

    # ── Unpack config ──
    num_classes       = int(config.get("num_classes", 7))
    class_names       = config.get("class_names", DEFAULT_CLASS_NAMES)
    class_labels      = config.get("class_labels", DEFAULT_CLASS_LABELS)
    output_dir        = config.get("output_dir", "models/CLIP/MultiMeditron-Gating-7class")
    num_epochs        = int(config.get("num_epochs", 20))
    lr                = float(config.get("lr", 1e-4))
    weight_decay      = float(config.get("weight_decay", 1e-4))
    batch_size        = int(config.get("batch_size", 32))
    num_workers       = int(config.get("num_workers", 4))
    max_samples       = int(config.get("max_samples_per_class", 0))
    val_split         = float(config.get("val_split", 0.1))
    test_split        = float(config.get("test_split", 0.1))
    save_every        = int(config.get("save_every_n_epochs", 5))
    seed              = int(config.get("seed", 42))
    use_wandb         = config.get("use_wandb", False)
    wandb_project     = config.get("wandb_project", "multimeditron-gating")
    wandb_run_name    = config.get("wandb_run_name", None)
    pretrained_bb     = config.get("pretrained_backbone", True)
    freeze_backbone   = config.get("freeze_backbone", True)
    unfreeze_after    = int(config.get("unfreeze_after_epoch", 0))
    resume_from       = config.get("resume_from", None)
    label_smoothing   = float(config.get("label_smoothing", 0.0))
    scheduler_type    = config.get("scheduler", "cosine")
    warmup_epochs     = int(config.get("warmup_epochs", 2))
    top_k             = int(config.get("top_k", 1))
    img_proc_path     = config.get("image_processor_path", "openai/clip-vit-base-patch32")
    grad_accum        = int(config.get("gradient_accumulation_steps", 1))

    assert len(class_names) == num_classes, \
        f"class_names ({len(class_names)}) != num_classes ({num_classes})"
    assert len(class_labels) == num_classes, \
        f"class_labels ({len(class_labels)}) != num_classes ({num_classes})"

    # YAML may parse keys as int or str
    dataset_class_map = {int(k): v for k, v in config.get("dataset_class_map", {}).items()}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── WandB ──
    if use_wandb and is_main(rank):
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # ── Build model ──
    log(rank, "\nBuilding GatingNetwork ...")

    gating_config = GatingNetworkConfig(
        num_classes=num_classes,
        top_k=top_k,
        image_processor_path=img_proc_path,
        class_names=class_names,
    )

    if resume_from and os.path.isdir(resume_from):
        log(rank, f"  Resuming from {resume_from}")
        model = GatingNetwork.from_pretrained(resume_from)
        model.config.class_names = class_names
        model.config.num_classes = num_classes
    else:
        model = GatingNetwork(gating_config)
        if pretrained_bb:
            log(rank, "  Loading ImageNet-pretrained ResNet50 backbone ...")
            # Download on rank 0 first, then barrier, to avoid filesystem races
            if is_main(rank):
                pretrained = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            if is_distributed:
                dist.barrier()
            if not is_main(rank):
                pretrained = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            pretrained_sd = pretrained.state_dict()
            model_sd = model.resnet.state_dict()
            for key in pretrained_sd:
                if key.startswith("fc."):
                    continue  # skip — different output dim
                if key in model_sd and pretrained_sd[key].shape == model_sd[key].shape:
                    model_sd[key] = pretrained_sd[key]
            model.resnet.load_state_dict(model_sd)
            del pretrained, pretrained_sd

    # Freeze backbone
    if freeze_backbone:
        for name, param in model.resnet.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(rank, f"  Total params: {total_p:,}  Trainable: {train_p:,}")

    model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if is_distributed else model

    # ── Build datasets ──
    log(rank, "\nLoading datasets ...")
    train_ds, val_ds, test_ds, num_classes, class_labels = build_datasets(
        {**config, "dataset_class_map": dataset_class_map},
        max_samples if max_samples > 0 else None,
        val_split, test_split, seed, rank,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = len(train_loader) * warmup_epochs // grad_accum

    if scheduler_type == "cosine" and total_steps > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_sched = LinearLR(optimizer, start_factor=0.01,
                                total_iters=max(warmup_steps, 1))
        cosine_sched = CosineAnnealingLR(optimizer,
                                         T_max=max(total_steps - warmup_steps, 1))
        scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                                 milestones=[warmup_steps])
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ── Training loop ──
    log(rank, f"\nStarting training for {num_epochs} epochs ...")
    if is_main(rank):
        os.makedirs(output_dir, exist_ok=True)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # Optionally unfreeze backbone
        if freeze_backbone and unfreeze_after > 0 and epoch == unfreeze_after:
            log(rank, f"\n  Unfreezing backbone at epoch {epoch}")
            for param in raw_model.resnet.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr * 0.1, weight_decay=weight_decay,
            )
            remaining = (num_epochs - epoch) * len(train_loader) // grad_accum
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 1))

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _, _ = model(images)
            loss = criterion(logits, labels) / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_type == "cosine":
                    scheduler.step()
                global_step += 1

            train_loss += loss.item() * grad_accum
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        if scheduler is not None and scheduler_type == "step":
            scheduler.step()

        train_loss /= max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits, _, _ = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for c in range(num_classes):
                    mask = labels == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += ((preds == labels) & mask).sum().item()

        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        # Aggregate across processes
        if is_distributed:
            metrics_t = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
            dist.all_reduce(metrics_t, op=dist.ReduceOp.AVG)
            train_loss, train_acc, val_loss, val_acc = metrics_t.tolist()

        current_lr = optimizer.param_groups[0]["lr"]

        if is_main(rank):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.2e}")

            for c in range(num_classes):
                c_acc = class_correct[c] / max(class_total[c], 1)
                lbl = class_labels[c] if c < len(class_labels) else f"class_{c}"
                bar = "█" * int(c_acc * 30)
                print(f"    {lbl:<15} {c_acc:.4f} ({class_correct[c]:>5}/{class_total[c]:<5}) {bar}")

            if use_wandb:
                import wandb
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "lr": current_lr,
                    "global_step": global_step,
                }
                for c in range(num_classes):
                    lbl = class_labels[c] if c < len(class_labels) else f"class_{c}"
                    log_dict[f"val/accuracy_{lbl}"] = class_correct[c] / max(class_total[c], 1)
                wandb.log(log_dict)

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(output_dir, "best")
                os.makedirs(best_path, exist_ok=True)
                raw_model.save_pretrained(best_path)
                print(f"  ★ New best model (val_acc={val_acc:.4f}) → {best_path}")

            # Periodic checkpoint
            if save_every > 0 and (epoch + 1) % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
                os.makedirs(ckpt_path, exist_ok=True)
                raw_model.save_pretrained(ckpt_path)
                print(f"  Checkpoint → {ckpt_path}")

    # ── Final save ──
    if is_main(rank):
        final_path = os.path.join(output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        raw_model.save_pretrained(final_path)
        print(f"\nTraining complete. Final model → {final_path}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

    # ── Held-out test evaluation ──
    if len(test_ds) > 0:
        log(rank, "\nRunning held-out test evaluation ...")
        test_sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed else None
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, sampler=test_sampler,
            num_workers=num_workers, pin_memory=True,
        )

        # Load best checkpoint for test eval
        best_path = os.path.join(output_dir, "best")
        if os.path.isdir(best_path):
            log(rank, f"  Loading best checkpoint from {best_path}")
            eval_model = GatingNetwork.from_pretrained(best_path)
            eval_model.to(device)
            eval_model.eval()
        else:
            eval_model = raw_model
            eval_model.eval()

        test_correct = 0
        test_total = 0
        class_correct_test = [0] * num_classes
        class_total_test = [0] * num_classes

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits, _, _ = eval_model(images)
                preds = logits.argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                for c in range(num_classes):
                    mask = labels == c
                    class_total_test[c] += mask.sum().item()
                    class_correct_test[c] += ((preds == labels) & mask).sum().item()

        test_acc = test_correct / max(test_total, 1)

        if is_main(rank):
            print(f"\n{'='*60}")
            print(f"  Held-out Test Results (n={test_total})")
            print(f"  Overall accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
            print(f"{'='*60}")
            for c in range(num_classes):
                c_acc = class_correct_test[c] / max(class_total_test[c], 1)
                lbl = class_labels[c] if c < len(class_labels) else f"class_{c}"
                bar = "█" * int(c_acc * 30)
                print(f"  {lbl:<15} {c_acc:.4f} ({class_correct_test[c]:>5}/{class_total_test[c]:<5}) {bar}")

            # Save test results to JSON
            import json
            test_results = {
                "overall_accuracy": test_acc,
                "total_samples": test_total,
                "per_class": {
                    class_labels[c] if c < len(class_labels) else f"class_{c}": {
                        "correct": class_correct_test[c],
                        "total": class_total_test[c],
                        "accuracy": class_correct_test[c] / max(class_total_test[c], 1),
                    }
                    for c in range(num_classes)
                },
            }
            results_path = os.path.join(output_dir, "test_results.json")
            with open(results_path, "w") as f:
                json.dump(test_results, f, indent=2)
            print(f"\n  Test results saved → {results_path}")

            if use_wandb:
                import wandb
                wandb.log({"test/overall_accuracy": test_acc})
                for c in range(num_classes):
                    lbl = class_labels[c] if c < len(class_labels) else f"class_{c}"
                    wandb.log({f"test/accuracy_{lbl}": class_correct_test[c] / max(class_total_test[c], 1)})

    if is_main(rank) and use_wandb:
        import wandb
        wandb.finish()

    cleanup_distributed()


# ─── Config loading & CLI ────────────────────────────────────────────────────

def load_config(config_path: Optional[str], cli_overrides: dict) -> dict:
    """Load YAML config and merge with CLI overrides (CLI wins)."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    for k, v in cli_overrides.items():
        if v is not None:
            config[k] = v

    return config


def parse_args():
    p = argparse.ArgumentParser(description="Train GatingNetwork (7-class image router)")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (e.g. scripts/config_gating.yaml)")

    # All config values can be overridden from CLI
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--max_samples_per_class", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--val_split", type=float, default=None)
    p.add_argument("--test_split", type=float, default=None,
                   help="Fraction of data held out as a separate test set (default: 0.1). Set 0 to disable.")
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--save_every_n_epochs", type=int, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--scheduler", type=str, default=None, choices=["cosine", "step", "none"])
    p.add_argument("--wandb", dest="use_wandb", action="store_true", default=None,
                   help="Enable wandb logging")
    p.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false", default=None,
                   help="Train full backbone (default: freeze)")
    p.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_after_epoch", type=int, default=None)
    p.add_argument("--pretrained_backbone", action="store_true", default=None)
    p.add_argument("--no_pretrained_backbone", dest="pretrained_backbone", action="store_false")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--class_names", type=str, default=None,
                   help="Comma-separated expert model paths (overrides config)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build CLI overrides (only non-None values)
    cli = {k: v for k, v in vars(args).items() if v is not None and k != "config"}

    # Handle --class_names comma-separated → list
    if "class_names" in cli and isinstance(cli["class_names"], str):
        cli["class_names"] = [s.strip() for s in cli["class_names"].split(",")]

    config = load_config(args.config, cli)

    # Apply defaults for anything not in config
    config.setdefault("num_classes", 7)
    config.setdefault("class_names", DEFAULT_CLASS_NAMES)
    config.setdefault("class_labels", DEFAULT_CLASS_LABELS)
    config.setdefault("output_dir", "models/CLIP/MultiMeditron-Gating-7class")
    config.setdefault("num_epochs", 20)
    config.setdefault("lr", 1e-4)
    config.setdefault("weight_decay", 1e-4)
    config.setdefault("batch_size", 32)
    config.setdefault("num_workers", 4)
    config.setdefault("max_samples_per_class", 0)
    config.setdefault("val_split", 0.1)
    config.setdefault("test_split", 0.1)
    config.setdefault("save_every_n_epochs", 5)
    config.setdefault("seed", 42)
    config.setdefault("use_wandb", False)
    config.setdefault("wandb_project", "multimeditron-gating")
    config.setdefault("pretrained_backbone", True)
    config.setdefault("freeze_backbone", True)
    config.setdefault("unfreeze_after_epoch", 0)
    config.setdefault("label_smoothing", 0.0)
    config.setdefault("scheduler", "cosine")
    config.setdefault("warmup_epochs", 2)
    config.setdefault("top_k", 1)
    config.setdefault("image_processor_path", "openai/clip-vit-base-patch32")
    config.setdefault("gradient_accumulation_steps", 1)

    if "dataset_class_map" not in config:
        print("ERROR: 'dataset_class_map' must be specified in the YAML config.")
        print("Example:")
        print("  dataset_class_map:")
        print("    0: ['/path/to/ct_arrow_dataset']")
        print("    1: ['/path/to/general_dataset1', '/path/to/general_dataset2']")
        sys.exit(1)

    train(config)
