"""
DeepShield AI — Main Training Script

Usage:
  python training/train.py --model efficientnet_b4 --dataset ff++ --epochs 20

Supports:
  - Transfer learning from ImageNet pretrained (timm)
  - Mixed precision training (AMP) for faster GPU training
  - Early stopping on validation AUC
  - Model checkpointing (best + last)
  - TensorBoard logging
  - Cross-validation option

Datasets supported (--dataset flag):
  ff++     = FaceForensics++ (place in data_dir/ff++/)
  celeb_df = Celeb-DF v2    (place in data_dir/celeb_df/)
  dfdc     = DFDC            (place in data_dir/dfdc/)
  wild     = WildDeepfake   (place in data_dir/wild/)
  all      = Unified (all datasets merged)
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.datasets.unified_dataset import UnifiedDeepfakeDataset
from training.models.efficientnet_deepfake import EfficientNetDeepfake
from training.models.xception_deepfake   import XceptionDeepfake
from training.evaluate                   import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepShield AI Training Pipeline")
    p.add_argument("--model",         default="efficientnet_b4",
                   choices=["efficientnet_b4", "efficientnet_b0", "xception", "vit_b16", "mesonet4"])
    p.add_argument("--dataset",       default="ff++",
                   choices=["ff++", "celeb_df", "dfdc", "wild", "all"])
    p.add_argument("--data_dir",      default="./data/datasets")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--val_split",     type=float, default=0.15)
    p.add_argument("--patience",      type=int,   default=5,   help="Early stopping patience")
    p.add_argument("--image_size",    type=int,   default=224)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--output_dir",    default="./models")
    p.add_argument("--resume",        default=None, help="Path to checkpoint to resume from")
    p.add_argument("--amp",           action="store_true", default=True, help="Mixed precision training")
    p.add_argument("--no_amp",        dest="amp", action="store_false")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> nn.Module:
    if name.startswith("efficientnet"):
        return EfficientNetDeepfake(variant=name, num_classes=2, pretrained=True)
    elif name == "xception":
        return XceptionDeepfake(num_classes=2, pretrained=True)
    else:
        # Use model_zoo for other architectures
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backend.detection.model_zoo import get_model
        return get_model(name, pretrained=True)


def train_epoch(model, loader, optimizer, criterion, scaler, device, amp=True):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}", flush=True)

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader.dataset), metrics


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f" DeepShield AI — Training Pipeline")
    print(f" Model:   {args.model}")
    print(f" Dataset: {args.dataset}")
    print(f" Device:  {device}")
    print(f" AMP:     {args.amp and torch.cuda.is_available()}")
    print(f"{'='*60}\n")

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_dataset = UnifiedDeepfakeDataset(
        data_dir   = args.data_dir,
        datasets   = [args.dataset] if args.dataset != "all" else ["ff++", "celeb_df", "wild"],
        image_size = args.image_size,
        augment    = True,
    )

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))

    val_ds.dataset.augment = False  # no augmentation during validation

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"Train samples: {train_size}  |  Val samples: {val_size}")
    print(f"Steps/epoch:   {len(train_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(args.model).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print(f"Resumed from: {args.resume}")

    # ── Optimiser + Scheduler ──────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler(enabled=args.amp and torch.cuda.is_available())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(f"runs/{args.model}_{args.dataset}_{datetime.now():%Y%m%d_%H%M}") \
                if TB_AVAILABLE else None

    # ── Training loop ──────────────────────────────────────────────────────────
    best_auc       = 0.0
    patience_count = 0
    history        = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler,
                                            device, amp=args.amp and torch.cuda.is_available())
        val_loss, val_metrics  = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        auc     = val_metrics["auc"]
        f1      = val_metrics["f1"]

        print(f"Epoch {epoch}/{args.epochs} | "
              f"loss={train_loss:.4f} | val_auc={auc:.4f} | val_f1={f1:.4f} | {elapsed:.0f}s",
              flush=True)

        if tb_writer:
            tb_writer.add_scalar("Loss/train",  train_loss, epoch)
            tb_writer.add_scalar("Loss/val",    val_loss,   epoch)
            tb_writer.add_scalar("AUC/val",     auc,        epoch)
            tb_writer.add_scalar("F1/val",      f1,         epoch)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "auc": auc, "f1": f1})

        # Save last checkpoint
        last_path = output_dir / f"{args.model}_last.pth"
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": auc, "history": history,
        }, last_path)

        # Save best checkpoint
        if auc > best_auc:
            best_auc = auc
            patience_count = 0
            best_path = output_dir / f"{args.model}_deepfake.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best AUC={auc:.4f} — saved to {best_path}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (patience={args.patience})")
                break

    # Save history
    (output_dir / f"{args.model}_history.json").write_text(json.dumps(history, indent=2))

    print(f"\n{'='*60}")
    print(f" Training complete! Best AUC = {best_auc:.4f}")
    print(f" Weights saved to: {output_dir}/{args.model}_deepfake.pth")
    print(f"{'='*60}")

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
