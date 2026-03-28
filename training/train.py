"""
KAVACH-AI Model Trainer
=======================
Fine-tunes any KAVACH-AI model from model_config.yaml.

Features:
  - HuggingFace + timm model loading
  - Staged layer freezing with warmup
  - Mixed precision (AMP) — auto-disabled on CPU
  - Gradient accumulation + gradient clipping
  - Cosine annealing LR scheduler with linear warmup
  - Best-checkpoint saving by val_auc
  - Early stopping (patience configurable)
  - Per-epoch: loss, accuracy, AUC-ROC, F1, precision, recall
  - TensorBoard + optional WandB logging
  - ONNX export after training
  - Test-set evaluation with full report

Usage:
  python training/train.py --model vit_primary
  python training/train.py --model efficientnet --epochs 20 --batch_size 16
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score,
                              classification_report)
import yaml
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import build_dataloaders

log = logging.getLogger('kavach.train')


# ── Reproducibility ────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ── Model loading ──────────────────────────────────────────────────
def load_model(cfg: dict, num_classes: int, device: str) -> nn.Module:
    """Load model from HuggingFace hub or timm depending on cfg['backbone']."""
    backbone = cfg['backbone']

    if backbone == 'hf':
        from transformers import AutoModelForImageClassification
        log.info(f"Loading HF model: {cfg['hf_model_id']}")
        model = AutoModelForImageClassification.from_pretrained(
            cfg['hf_model_id'],
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        return model.to(device)

    elif backbone == 'timm':
        import timm
        timm_id  = cfg['timm_model_id']
        pretrained = cfg.get('pretrained', True)
        log.info(f'Loading timm model: {timm_id}  pretrained={pretrained}')
        model = timm.create_model(
            timm_id,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=cfg.get('dropout', 0.0),
        )
        return model.to(device)

    else:
        raise ValueError(f"Unknown backbone: {backbone}. Use 'hf' or 'timm'.")


# ── Layer freezing ─────────────────────────────────────────────────
def apply_freeze_strategy(model: nn.Module, strategy: str,
                           backbone: str) -> None:
    """
    Freeze model layers according to strategy.
    Called at warmup start; reversed at warmup end.
    """
    if strategy == 'none':
        return

    if strategy == 'staged':
        # Freeze everything except the classifier head
        for name, param in model.named_parameters():
            is_head = any(k in name for k in
                          ['classifier', 'head', 'fc', 'last_linear'])
            param.requires_grad = is_head
        log.info('Freeze: all layers except classifier head')
        return

    if strategy == 'head_plus_last2' and backbone == 'hf':
        # ViT: freeze all except classifier + last 2 encoder blocks
        for name, param in model.named_parameters():
            is_head   = 'classifier' in name
            is_last2  = any(f'encoder.layer.{i}' in name
                            for i in [10, 11])
            param.requires_grad = is_head or is_last2
        log.info('Freeze: ViT all except head + encoder layers 10,11')
        return

    if strategy == 'head_plus_last1' and backbone == 'hf':
        for name, param in model.named_parameters():
            is_head  = 'classifier' in name
            is_last1 = 'encoder.layer.11' in name
            param.requires_grad = is_head or is_last1
        log.info('Freeze: ViT all except head + encoder layer 11')
        return

    log.warning(f'Unknown freeze strategy: {strategy} — no freezing applied')


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters after warmup epochs."""
    for param in model.parameters():
        param.requires_grad = True
    log.info('All layers unfrozen — full fine-tuning begins')


# ── Scheduler ──────────────────────────────────────────────────────
def build_scheduler(optimizer, cfg: dict, n_batches: int):
    """Cosine annealing with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = cfg.get('warmup_epochs', 3) * n_batches
    total_steps  = cfg.get('epochs', 30) * n_batches

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ── Loss ───────────────────────────────────────────────────────────
def build_criterion(label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# ── One epoch ──────────────────────────────────────────────────────
def run_epoch(
    model, loader, criterion, optimizer, scaler,
    scheduler, device, cfg, is_train: bool,
) -> dict:
    model.train() if is_train else model.eval()
    amp_enabled = cfg.get('amp_enabled', True) and device.startswith('cuda')
    grad_accum  = cfg.get('grad_accumulation', 1)
    grad_clip   = cfg.get('grad_clip', 1.0)
    backbone    = cfg.get('backbone', 'timm')

    total_loss, n_correct, n_total = 0.0, 0, 0
    all_probs, all_labels = [], []

    if is_train:
        optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            if backbone == 'hf':
                outputs = model(pixel_values=images)
                logits  = outputs.logits
            else:
                logits = model(images)
            loss = criterion(logits, labels) / grad_accum

        if is_train:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        # Metrics accumulation
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            n_correct  += (preds == labels).sum().item()
            n_total    += labels.size(0)
            total_loss += loss.item() * grad_accum * labels.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= 0.5).astype(int)

    return {
        'loss':      total_loss / n_total,
        'accuracy':  n_correct  / n_total,
        'auc':       roc_auc_score(all_labels, all_probs),
        'f1':        f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall':    recall_score(all_labels, all_preds, zero_division=0),
    }


# ── Main training loop ─────────────────────────────────────────────
def train(cfg_path: str, model_key: str, overrides: dict) -> Path:
    """
    Main entry point. Loads config, trains model, saves checkpoint.
    Returns path to best checkpoint file.
    """
    # Load config
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    shared = full_cfg.get('shared', {})
    cfg    = {**shared, **full_cfg[model_key], **overrides}

    # Setup
    set_seed(cfg.get('seed', 42))
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_enabled = cfg.get('amp_enabled', True) and device == 'cuda'

    ckpt_dir = Path(cfg['checkpoint_dir']) / cfg['model_name']
    log_dir  = Path(cfg['log_dir'])  / cfg['model_name']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / 'train.log')],
    )
    log.info(f'Training {model_key} on {device}')
    log.info(f'Config: {json.dumps(cfg, default=str, indent=2)}')

    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))

    # WandB (optional)
    wandb_run = None
    if cfg.get('wandb_enabled') and os.environ.get('WANDB_API_KEY'):
        import wandb
        wandb_run = wandb.init(
            project=cfg.get('wandb_project', 'kavach-ai'),
            name=cfg['model_name'],
            config=cfg,
        )
        
    # MLflow Setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg.get("mlflow_experiment", "kavach-ai-training"))
    
    with mlflow.start_run(run_name=cfg['model_name']) as run:
        mlflow.log_params(cfg)

        # Data
        loaders = build_dataloaders(
            cfg['manifest_path'],
            batch_size=cfg.get('batch_size', 32),
            num_workers=cfg.get('num_workers', 4),
            balanced_sampling=True,
        )

    # Model
    model = load_model(cfg, num_classes=cfg.get('num_classes', 2), device=device)

    # Freeze strategy for warmup
    apply_freeze_strategy(model,
                          cfg.get('freeze_strategy', 'none'),
                          cfg.get('backbone', 'timm'))

    # Optimizer: separate LR for head vs backbone
    head_params     = [p for n,p in model.named_parameters()
                       if p.requires_grad and any(
                           k in n for k in ['classifier','head','fc','last_linear']
                       )]
    backbone_params = [p for n,p in model.named_parameters()
                       if p.requires_grad and not any(
                           k in n for k in ['classifier','head','fc','last_linear']
                       )]
    optimizer = torch.optim.AdamW([
        {'params': head_params,     'lr': cfg.get('head_lr', 1e-3)},
        {'params': backbone_params, 'lr': cfg.get('backbone_lr', 1e-5)},
    ], weight_decay=cfg.get('weight_decay', 1e-4))

    scaler    = GradScaler(enabled=amp_enabled)
    criterion = build_criterion(cfg.get('label_smoothing', 0.0))
    scheduler = build_scheduler(optimizer, cfg, len(loaders['train']))

    # Training state
    best_auc    = 0.0
    best_ckpt   = None
    patience    = cfg.get('early_stop_patience', 5)
    no_improve  = 0
    epochs      = cfg.get('epochs', 30)
    warmup_ep   = cfg.get('warmup_epochs', 3)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if epoch == warmup_ep + 1:
            unfreeze_all(model)
            log.info(f'Epoch {epoch}: backbone unfrozen')

        # Train
        train_metrics = run_epoch(
            model, loaders['train'], criterion, optimizer,
            scaler, scheduler, device, cfg, is_train=True,
        )

        # Validate
        with torch.no_grad():
            val_metrics = run_epoch(
                model, loaders['val'], criterion, optimizer,
                scaler, None, device, cfg, is_train=False,
            )

        elapsed = time.time() - t0
        log.info(
            f'Epoch {epoch:>3}/{epochs}  '
            f'train_loss={train_metrics["loss"]:.4f}  '
            f'val_auc={val_metrics["auc"]:.4f}  '
            f'val_acc={val_metrics["accuracy"]:.4f}  '
            f'({elapsed:.0f}s)'
        )

        # TensorBoard logging
        for phase, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for k, v in metrics.items():
                writer.add_scalar(f'{phase}/{k}', v, epoch)
        writer.add_scalar('lr/head',     optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('lr/backbone', optimizer.param_groups[1]['lr'], epoch)

        # WandB logging
        if wandb_run:
            wandb_run.log({
                **{f'train_{k}': v for k,v in train_metrics.items()},
                **{f'val_{k}':   v for k,v in val_metrics.items()},
                'epoch': epoch,
            })
            
        # MLflow metrics logging
        mlflow.log_metrics({f'train_{k}': v for k,v in train_metrics.items()}, step=epoch)
        mlflow.log_metrics({f'val_{k}': v for k,v in val_metrics.items()}, step=epoch)

        # Save best checkpoint
        val_auc = val_metrics['auc']
        if best_ckpt is None or val_auc > best_auc:
            best_auc  = val_auc
            best_ckpt = ckpt_dir / f'{cfg["model_name"]}_best_auc{val_auc:.4f}.pt'
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'val_auc':     val_auc,
                'cfg':         cfg,
            }, best_ckpt)
            log.info(f'  ✓ Best checkpoint saved: {best_ckpt.name}')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f'Early stopping: no improvement for {patience} epochs')
                break

    writer.close()

    # ── Test evaluation ──────────────────────────────────────────────
    log.info('Loading best checkpoint for test evaluation...')
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    with torch.no_grad():
        test_metrics = run_epoch(
            model, loaders['test'], criterion, optimizer,
            scaler, None, device, cfg, is_train=False,
        )

    log.info('=== TEST SET RESULTS ===')
    for k, v in test_metrics.items():
        log.info(f'  {k:<12}: {v:.4f}')

    # Save results JSON
    results_path = ckpt_dir / f'{cfg["model_name"]}_test_results.json'
    with open(results_path, 'w') as f:
        json.dump({'test': test_metrics, 'best_val_auc': best_auc,
                   'model': cfg['model_name']}, f, indent=2)

        # ── ONNX export ──────────────────────────────────────────────────
        onnx_path = ckpt_dir / f'{cfg["model_name"]}.onnx'
        export_onnx(model, cfg, onnx_path, device)
        
        # Log artifacts to MLflow
        if best_ckpt and best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints")
        if onnx_path.exists():
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx_models")
        if results_path.exists():
            mlflow.log_artifact(str(results_path), artifact_path="results")

        if wandb_run:
            wandb_run.log({f'test_{k}': v for k, v in test_metrics.items()})
            wandb_run.finish()

        log.info(f'Training complete. Best val AUC: {best_auc:.4f}')
        log.info(f'Checkpoint: {best_ckpt}')
        log.info(f'ONNX model: {onnx_path}')
        return best_ckpt


# ── ONNX export ────────────────────────────────────────────────────
def export_onnx(model: nn.Module, cfg: dict,
                out_path: Path, device: str) -> None:
    """Export model to ONNX for fast CPU/GPU inference."""
    model.eval()
    img_size  = cfg.get('img_size', 224)
    backbone  = cfg.get('backbone', 'timm')
    dummy     = torch.randn(1, 3, img_size, img_size).to(device)

    try:
        if backbone == 'hf':
            # HF models need pixel_values kwarg wrapper
            class HFWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x): return self.m(pixel_values=x).logits
            export_model = HFWrapper(model)
        else:
            export_model = model

        torch.onnx.export(
            export_model, dummy, str(out_path),
            input_names=['input'],
            output_names=['logits'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'logits': {0: 'batch_size'}},
            opset_version=17,
            do_constant_folding=True,
        )
        log.info(f'ONNX export: {out_path}')
        # Verify
        import onnx
        onnx.checker.check_model(str(out_path))
        log.info('ONNX model check: PASSED')
    except Exception as e:
        log.warning(f'ONNX export failed: {e} — skipping')


# ── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='KAVACH-AI Model Trainer')
    parser.add_argument('--model',      required=True,
                        choices=['vit_primary','vit_secondary','efficientnet',
                                 'xception','convnext'],
                        help='Model key from model_config.yaml')
    parser.add_argument('--config',     default='training/model_config.yaml')
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--device',     default=None,
                        choices=['cpu','cuda'])
    args = parser.parse_args()

    overrides = {k: v for k, v in {
        'epochs':     args.epochs,
        'batch_size': args.batch_size,
        'head_lr':    args.lr,
    }.items() if v is not None}

    train(args.config, args.model, overrides)


if __name__ == '__main__':
    main()
