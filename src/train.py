import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from src.dataset import parse_selection_files, build_dataset_clips, WhaleDataset, CLASS_MAPPING
from src.model import WhaleSEDModel
from src.losses import MultiLabelFocalLoss

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    """
    Standard training epoch without domain adaptation.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for i, (specs, labels) in enumerate(loader):
        specs = specs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if mixup_alpha > 0.0 and np.random.rand() < 0.5:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            permutation = torch.randperm(specs.size(0))
            mixed_specs = lam * specs + (1.0 - lam) * specs[permutation]
            mixed_labels = lam * labels + (1.0 - lam) * labels[permutation]
            
            # Normal forward pass (return_domain defaults to False)
            probs = model(mixed_specs)
            loss = criterion(probs, mixed_labels)
        else:
            probs = model(specs)
            loss = criterion(probs, labels)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * specs.size(0)
        
    epoch_loss = running_loss / len(loader.dataset)
    elapsed = time.time() - start_time
    return epoch_loss, elapsed

def train_epoch_dann(model, source_loader, target_loader, optimizer, species_criterion, domain_criterion, device, mixup_alpha=0.0, dann_alpha=1.0):
    """
    Domain-Adversarial Neural Network (DANN) training epoch.
    Pulls batch pairs from source (labeled calls & domain 0) and target (unlabeled calls & domain 1).
    """
    model.train()
    running_loss = 0.0
    running_species_loss = 0.0
    running_domain_loss = 0.0
    
    start_time = time.time()
    target_iter = iter(target_loader)
    
    for i, (source_specs, source_labels) in enumerate(source_loader):
        # Fetch target batch, cycle target_loader if it runs out of items
        try:
            target_specs, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_specs, _ = next(target_iter)
            
        B_s = source_specs.size(0)
        B_t = target_specs.size(0)
        
        # Domain target variables: Source = 0, Target = 1
        source_domain_targets = torch.zeros(B_s, 1).to(device)
        target_domain_targets = torch.ones(B_t, 1).to(device)
        
        # Load inputs on target hardware
        source_specs = source_specs.to(device)
        source_labels = source_labels.to(device)
        target_specs = target_specs.to(device)
        
        optimizer.zero_grad()
        
        # --- 1. Source Forward Pass (Classify species + domain) ---
        if mixup_alpha > 0.0 and np.random.rand() < 0.5:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            permutation = torch.randperm(B_s)
            mixed_specs = lam * source_specs + (1.0 - lam) * source_specs[permutation]
            mixed_labels = lam * source_labels + (1.0 - lam) * source_labels[permutation]
            
            source_probs, source_domain = model(mixed_specs, alpha=dann_alpha, return_domain=True)
            species_loss = species_criterion(source_probs, mixed_labels)
        else:
            source_probs, source_domain = model(source_specs, alpha=dann_alpha, return_domain=True)
            species_loss = species_criterion(source_probs, source_labels)
            
        # --- 2. Target Forward Pass (Classify domain only) ---
        _, target_domain = model(target_specs, alpha=dann_alpha, return_domain=True)
        
        # --- 3. Compute joint losses ---
        domain_loss_s = domain_criterion(source_domain, source_domain_targets)
        domain_loss_t = domain_criterion(target_domain, target_domain_targets)
        domain_loss = (domain_loss_s + domain_loss_t) / 2.0
        
        # GRL reverses domain classification gradients, penalizing feature domain specificity
        loss = species_loss + domain_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * B_s
        running_species_loss += species_loss.item() * B_s
        running_domain_loss += domain_loss.item() * B_s
        
    epoch_loss = running_loss / len(source_loader.dataset)
    epoch_species_loss = running_species_loss / len(source_loader.dataset)
    epoch_domain_loss = running_domain_loss / len(source_loader.dataset)
    elapsed = time.time() - start_time
    
    return epoch_loss, epoch_species_loss, epoch_domain_loss, elapsed

def validate(model, loader, criterion, device, class_thresholds=None):
    model.eval()
    running_loss = 0.0
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            labels = labels.to(device)
            
            probs = model(specs)
            loss = criterion(probs, labels)
            running_loss += loss.item() * specs.size(0)
            
            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    
    all_targets = np.concatenate(all_targets, axis=0) # shape: (N, T, num_classes)
    all_preds = np.concatenate(all_preds, axis=0)     # shape: (N, T, num_classes)
    
    N, T, C = all_targets.shape
    flat_targets = all_targets.reshape(-1, C)
    flat_preds = all_preds.reshape(-1, C)
    
    # Evaluate baseline global 0.5 threshold
    preds_binary_05 = (flat_preds >= 0.5).astype(float)
    f1_macro_05 = f1_score(flat_targets, preds_binary_05, average='macro', zero_division=0)
    
    # Search for optimal thresholds on validation set (if none provided)
    opt_thresholds = {}
    if class_thresholds is not None:
        opt_thresholds = class_thresholds
    else:
        threshold_range = np.arange(0.05, 0.96, 0.05)
        for cls_name, cls_idx in CLASS_MAPPING.items():
            best_th = 0.5
            best_f1 = 0.0
            y_true = flat_targets[:, cls_idx]
            y_pred = flat_preds[:, cls_idx]
            
            for th in threshold_range:
                f1 = f1_score(y_true, (y_pred >= th).astype(float), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th
            opt_thresholds[cls_name] = float(best_th)
            
    # Calculate binary predictions using target thresholds
    preds_binary_opt = np.zeros_like(flat_preds)
    class_f1s = {}
    for cls_name, cls_idx in CLASS_MAPPING.items():
        th = opt_thresholds[cls_name]
        preds_binary_opt[:, cls_idx] = (flat_preds[:, cls_idx] >= th).astype(float)
        class_f1s[cls_name] = f1_score(flat_targets[:, cls_idx], preds_binary_opt[:, cls_idx], zero_division=0)
        
    f1_macro_opt = f1_score(flat_targets, preds_binary_opt, average='macro', zero_division=0)
    
    # Calculate AUC-ROC
    try:
        auc_macro = roc_auc_score(flat_targets, flat_preds, average='macro')
    except ValueError:
        auc_macro = 0.5
        
    return val_loss, f1_macro_05, f1_macro_opt, auc_macro, class_f1s, opt_thresholds

def main():
    parser = argparse.ArgumentParser(description="Train Whale Vocalization SED Model with Advanced Pipeline & DANN")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory of training dataset")
    parser.add_argument("--train_name", type=str, required=True, help="Name of training dataset (casey2014 or Greenwich64S2015)")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory of validation dataset")
    parser.add_argument("--val_name", type=str, required=True, help="Name of validation dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/best_model.pth", help="Path to save the best model")
    parser.add_argument("--neg_multiplier", type=float, default=1.0, help="Ratio of negative clips to positive clips")
    parser.add_argument("--subsample_train", type=float, default=1.0, help="Fraction to subsample training clips (for fast tests)")
    parser.add_argument("--subsample_val", type=float, default=1.0, help="Fraction to subsample validation clips (for fast tests)")
    
    # Advanced Pipeline Configurations
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal"], help="Loss function type")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma focusing parameter for Focal Loss")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Alpha for Mixup augmentation (<= 0.0 to disable)")
    parser.add_argument("--spec_augment", action="store_true", help="Enable SpecAugment frequency & time masking")
    
    # DANN Unsupervised Domain Adaptation Configurations
    parser.add_argument("--dann", action="store_true", help="Enable Domain-Adversarial Neural Network (DANN) training")
    parser.add_argument("--dann_alpha", type=float, default=1.0, help="GRL scaling coefficient alpha")
    parser.add_argument("--target_dir", type=str, default=None, help="Directory of target domain dataset (defaults to val_dir)")
    parser.add_argument("--target_name", type=str, default=None, help="Name of target domain dataset")
    
    args = parser.parse_args()
    
    # Make sure output directories exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Pipeline config: Loss={args.loss_type} | Mixup={args.mixup_alpha} | SpecAugment={args.spec_augment}")
    if args.dann:
        print(f"Domain Adaptation: DANN=True | GRL Alpha={args.dann_alpha}")
    
    # 1. Load and parse dataset selections
    print(f"Loading training data from: {args.train_dir}")
    train_anns = parse_selection_files(args.train_dir, args.train_name)
    train_clips = build_dataset_clips(args.train_dir, args.train_name, train_anns, neg_multiplier=args.neg_multiplier)
    
    print(f"Loading validation data from: {args.val_dir}")
    val_anns = parse_selection_files(args.val_dir, args.val_name)
    val_clips = build_dataset_clips(args.val_dir, args.val_name, val_anns, neg_multiplier=args.neg_multiplier)
    
    # Subsample training data if requested
    if args.subsample_train < 1.0:
        np.random.shuffle(train_clips)
        num_keep = int(len(train_clips) * args.subsample_train)
        train_clips = train_clips[:num_keep]
        print(f"Subsampled training clips to: {len(train_clips)}")
        
    # Subsample validation data if requested
    if args.subsample_val < 1.0:
        np.random.shuffle(val_clips)
        num_keep = int(len(val_clips) * args.subsample_val)
        val_clips = val_clips[:num_keep]
        print(f"Subsampled validation clips to: {len(val_clips)}")
        
    print(f"Total training clips: {len(train_clips)}")
    print(f"Total validation clips: {len(val_clips)}")
    
    # Create datasets and dataloaders
    train_dataset = WhaleDataset(train_clips, augment=True, spec_augment=args.spec_augment)
    val_dataset = WhaleDataset(val_clips, augment=False, spec_augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Setup DANN target domain resources if enabled
    if args.dann:
        target_dir = args.target_dir if args.target_dir else args.val_dir
        target_name = args.target_name if args.target_name else args.val_name
        
        print(f"Loading target domain training data from: {target_dir}")
        target_anns = parse_selection_files(target_dir, target_name)
        target_clips = build_dataset_clips(target_dir, target_name, target_anns, neg_multiplier=args.neg_multiplier)
        
        if args.subsample_val < 1.0:
            np.random.shuffle(target_clips)
            num_keep = int(len(target_clips) * args.subsample_val)
            target_clips = target_clips[:num_keep]
            print(f"Subsampled target training clips to: {len(target_clips)}")
            
        target_dataset = WhaleDataset(target_clips, augment=True, spec_augment=args.spec_augment)
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        domain_criterion = nn.BCELoss()
    else:
        target_loader = None
        domain_criterion = None
        
    # Initialize Model
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING)).to(device)
    
    # Select criterion
    if args.loss_type == "focal":
        criterion = MultiLabelFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.BCELoss()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_f1 = -1.0
    best_epoch = 0
    
    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        # Train (running standard or DANN adversarial loops)
        if args.dann:
            train_loss, train_species_loss, train_domain_loss, train_time = train_epoch_dann(
                model=model,
                source_loader=train_loader,
                target_loader=target_loader,
                optimizer=optimizer,
                species_criterion=criterion,
                domain_criterion=domain_criterion,
                device=device,
                mixup_alpha=args.mixup_alpha,
                dann_alpha=args.dann_alpha
            )
        else:
            train_loss, train_time = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                mixup_alpha=args.mixup_alpha
            )
            train_species_loss = train_loss
            train_domain_loss = 0.0
            
        # Validate (running threshold search dynamically)
        val_loss, val_f1_05, val_f1_opt, val_auc, class_f1s, opt_thresholds = validate(model, val_loader, criterion, device)
        
        # Step LR scheduler
        scheduler.step()
        
        # Log training details
        if args.dann:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} (Species: {train_species_loss:.4f}, Domain: {train_domain_loss:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1 (Macro 0.5): {val_f1_05:.4f} | "
                  f"Val F1 (Macro Opt): {val_f1_opt:.4f} | "
                  f"Val AUC (Macro): {val_auc:.4f} | "
                  f"Time: {train_time:.1f}s")
        else:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1 (Macro 0.5): {val_f1_05:.4f} | "
                  f"Val F1 (Macro Opt): {val_f1_opt:.4f} | "
                  f"Val AUC (Macro): {val_auc:.4f} | "
                  f"Time: {train_time:.1f}s")
                  
        # Print class F1s for classes that have positive instances
        f1_strs = []
        for cls_name, f1 in class_f1s.items():
            if f1 > 0:
                f1_strs.append(f"{cls_name} (Th={opt_thresholds[cls_name]:.2f}): {f1:.3f}")
        if f1_strs:
            print("  Active Class F1s: " + ", ".join(f1_strs))
            
        # Save best model checkpoint based on optimized (Opt) Val F1 score
        if val_f1_opt > best_val_f1:
            best_val_f1 = val_f1_opt
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
            
            # Save threshold config file
            thresholds_path = args.save_path.replace(".pth", "_thresholds.json")
            with open(thresholds_path, 'w', encoding='utf-8') as f:
                json.dump(opt_thresholds, f, indent=4)
                
            print(f"  --> Saved new best checkpoint to {args.save_path} and thresholds to {thresholds_path} (Opt F1: {val_f1_opt:.4f})")
            
    print(f"\nTraining completed! Best Epoch: {best_epoch} with Val F1 (Macro Opt): {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
