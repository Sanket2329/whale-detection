import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

try:
    import yaml
except ImportError:
    yaml = None

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        elif yaml is not None:
            return yaml.safe_load(f)
        else:
            # Fallback if PyYAML is not installed: try JSON parse
            return json.load(f)

from src.data.dataset import WhaleClipDataset, build_clips_from_manifest, CLASS_NAMES
from src.models.crnn import WhaleCRNN, WhaleCRNNDANN
from src.training.losses import FocalLoss
from src.training.checkpoint import save_checkpoint
from src.evaluation.threshold import find_best_thresholds_per_class

def parse_args():
    parser = argparse.ArgumentParser(description="Train Whale SED model (CRNN / DANN)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Compute device (mps, cuda, cpu)")
    parser.add_argument("--max_train_clips", type=int, default=None, help="Subsample train clips for rapid validation")
    parser.add_argument("--max_val_clips", type=int, default=None, help="Subsample val clips for rapid validation")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    return parser.parse_args()

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            
            logits = model(specs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * specs.size(0)
            
            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    val_loss = total_loss / len(val_loader.dataset)
    targets_concat = np.concatenate(all_targets, axis=0) # (N, 118, 8)
    probs_concat = np.concatenate(all_probs, axis=0)     # (N, 118, 8)
    
    N, T, C = targets_concat.shape
    flat_targets = targets_concat.reshape(-1, C)
    flat_probs = probs_concat.reshape(-1, C)
    
    # Baseline 0.5 threshold
    preds_05 = (flat_probs >= 0.5).astype(np.int32)
    macro_f1_05 = f1_score(flat_targets, preds_05, average="macro", zero_division=0)
    
    # Class-wise threshold optimization
    opt_thresholds, class_f1s, macro_f1_opt = find_best_thresholds_per_class(
        targets=flat_targets,
        probabilities=flat_probs,
        class_names=CLASS_NAMES
    )
    
    # Audit for uncalibrated classes
    uncalibrated = [c for c in CLASS_NAMES if np.sum(flat_targets[:, CLASS_NAMES.index(c)]) == 0]
    if uncalibrated:
        print(f"   [Validation Warning] No positive frames for: {uncalibrated}. Threshold defaulted to 0.5.")
    
    return val_loss, macro_f1_05, macro_f1_opt, opt_thresholds, class_f1s

def main():
    args = parse_args()
    cfg = load_config(args.config)
        
    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
    print(f"=== Starting Experiment: {cfg['experiment']['name']} ===")
    print(f"Device: {device} | Seed: {seed}")
    
    # 1. Build Clip Datasets from Independent Manifests
    data_cfg = cfg["data"]
    print("Building source training clips...")
    source_train_clips = build_clips_from_manifest(
        manifest_csv=data_cfg["source_train_manifest"],
        annotation_dir="data/casey2014",
        labelled=True,
        clip_duration=data_cfg.get("clip_duration", 30),
        stride=15.0,
        max_clips=args.max_train_clips,
        seed=seed
    )
    
    print("Building target validation clips...")
    target_val_clips = build_clips_from_manifest(
        manifest_csv=data_cfg["target_val_manifest"],
        annotation_dir="data/Greenwich64S2015",
        labelled=True,
        clip_duration=data_cfg.get("clip_duration", 30),
        stride=15.0,
        max_clips=args.max_val_clips,
        seed=seed
    )
    
    source_train_ds = WhaleClipDataset(source_train_clips, labelled=True)
    target_val_ds = WhaleClipDataset(target_val_clips, labelled=True)
    
    train_cfg = cfg["training"]
    batch_size = train_cfg.get("batch_size", 64)
    source_loader = DataLoader(source_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(target_val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded: {len(source_train_ds)} source train clips | {len(target_val_ds)} target val clips")
    
    # 2. Check for Domain Adaptation
    da_cfg = train_cfg.get("domain_adaptation", {})
    is_dann = da_cfg.get("enabled", False)
    target_loader = None
    
    if is_dann:
        print("Domain Adaptation enabled. Building unlabelled target adaptation clips...")
        target_adapt_clips = build_clips_from_manifest(
            manifest_csv=data_cfg["target_adapt_manifest"],
            annotation_dir=None,
            labelled=False,
            clip_duration=data_cfg.get("clip_duration", 30),
            stride=15.0,
            max_clips=args.max_train_clips,
            seed=seed
        )
        target_adapt_ds = WhaleClipDataset(target_adapt_clips, labelled=False)
        target_loader = DataLoader(target_adapt_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        print(f"Loaded: {len(target_adapt_ds)} unlabelled target adaptation clips")
        model = WhaleCRNNDANN(num_classes=data_cfg.get("num_classes", 8)).to(device)
    else:
        model = WhaleCRNN(num_classes=data_cfg.get("num_classes", 8)).to(device)
        
    # 3. Setup Loss and Optimizer
    loss_cfg = train_cfg.get("loss", {})
    if loss_cfg.get("type") == "focal":
        species_criterion = FocalLoss(alpha=loss_cfg.get("alpha", 0.25), gamma=loss_cfg.get("gamma", 2.0))
    else:
        species_criterion = nn.BCEWithLogitsLoss()
        
    domain_criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-4)
    )
    
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = -1.0
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoints_dir, f"{cfg['experiment']['name']}_best.pt")
    
    print("\n--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_species_loss = 0.0
        train_domain_loss = 0.0
        
        target_iter = iter(target_loader) if is_dann else None
        
        # Calculate gradual GRL alpha if DANN is active: 2.0 / (1.0 + exp(-10 * p)) - 1.0
        p = float(epoch - 1) / float(max(1, epochs))
        grl_alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0 if is_dann else 0.0
        
        for batch_idx, (src_specs, src_labels) in enumerate(source_loader):
            src_specs = src_specs.to(device)
            src_labels = src_labels.to(device)
            B_s = src_specs.size(0)
            
            optimizer.zero_grad()
            
            if is_dann:
                try:
                    tgt_specs = next(target_iter)
                except (StopIteration, TypeError):
                    target_iter = iter(target_loader)
                    tgt_specs = next(target_iter)
                    
                tgt_specs = tgt_specs.to(device)
                B_t = tgt_specs.size(0)
                
                # Forward source (species logits + domain logits)
                src_logits, src_domain = model(src_specs, alpha=grl_alpha, return_domain=True)
                # Forward target (domain logits only)
                _, tgt_domain = model(tgt_specs, alpha=grl_alpha, return_domain=True)
                
                loss_species = species_criterion(src_logits, src_labels)
                
                src_dom_targets = torch.zeros(B_s, 1, device=device)
                tgt_dom_targets = torch.ones(B_t, 1, device=device)
                loss_dom_s = domain_criterion(src_domain, src_dom_targets)
                loss_dom_t = domain_criterion(tgt_domain, tgt_dom_targets)
                loss_domain = 0.5 * (loss_dom_s + loss_dom_t)
                
                total_loss = loss_species + loss_domain
                total_loss.backward()
                optimizer.step()
                
                train_species_loss += loss_species.item()
                train_domain_loss += loss_domain.item()
            else:
                src_logits = model(src_specs)
                loss_species = species_criterion(src_logits, src_labels)
                loss_species.backward()
                optimizer.step()
                train_species_loss += loss_species.item()
                
        scheduler.step()
        
        avg_species_loss = train_species_loss / len(source_loader)
        avg_domain_loss = train_domain_loss / len(source_loader) if is_dann else 0.0
        
        # Validation on target validation partition
        val_loss, f1_05, f1_opt, opt_th, class_f1s = validate(model, val_loader, species_criterion, device)
        
        f1_summary = ", ".join([f"{k[:6]}:{v:.3f}" for k, v in class_f1s.items() if v > 0.0]) or "None"
        dann_info = f" | DomLoss: {avg_domain_loss:.4f} (α={grl_alpha:.2f})" if is_dann else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | TrainSpecLoss: {avg_species_loss:.4f}{dann_info} | ValLoss: {val_loss:.4f} | Val F1 (0.5): {f1_05:.4f} | Val F1 (Opt): {f1_opt:.4f}")
        print(f"   Active Classes F1: {f1_summary}")
        
        if f1_opt > best_val_f1:
            best_val_f1 = f1_opt
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_f1,
                config=cfg,
                seed=seed,
                class_thresholds=opt_th
            )
            print(f"   --> Saved new best checkpoint to {best_ckpt_path} (Opt F1: {best_val_f1:.4f})")
            
    print(f"\nTraining Complete. Best Validation F1 (Opt): {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
