import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from src.dataset import parse_selection_files, build_dataset_clips, WhaleDataset, CLASS_MAPPING
from src.model import WhaleSEDModel

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for i, (specs, labels) in enumerate(loader):
        # specs shape: (B, 1, F, T), labels shape: (B, T, num_classes)
        specs = specs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        probs = model(specs) # shape: (B, T, num_classes)
        
        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * specs.size(0)
        
    epoch_loss = running_loss / len(loader.dataset)
    elapsed = time.time() - start_time
    return epoch_loss, elapsed

def validate(model, loader, criterion, device):
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
    
    # Flatten across batch and time dimensions for frame-level evaluation
    N, T, C = all_targets.shape
    flat_targets = all_targets.reshape(-1, C)
    flat_preds = all_preds.reshape(-1, C)
    
    # Calculate macro F1-score (threshold at 0.5)
    preds_binary = (flat_preds >= 0.5).astype(float)
    f1_macro = f1_score(flat_targets, preds_binary, average='macro', zero_division=0)
    
    # Calculate individual class F1-scores
    class_f1s = {}
    for cls_name, cls_idx in CLASS_MAPPING.items():
        class_f1s[cls_name] = f1_score(flat_targets[:, cls_idx], preds_binary[:, cls_idx], zero_division=0)
        
    # Calculate AUC-ROC
    try:
        auc_macro = roc_auc_score(flat_targets, flat_preds, average='macro')
    except ValueError:
        auc_macro = 0.5 # if no positive frames for some classes in validation
        
    return val_loss, f1_macro, auc_macro, class_f1s

def main():
    parser = argparse.ArgumentParser(description="Train Whale Vocalization SED Model")
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
    
    args = parser.parse_args()
    
    # Make sure output directories exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
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
    train_dataset = WhaleDataset(train_clips, augment=True)
    val_dataset = WhaleDataset(val_clips, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize Model, Loss, Optimizer
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_f1 = -1.0
    best_epoch = 0
    
    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_time = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_f1, val_auc, class_f1s = validate(model, val_loader, criterion, device)
        
        # Step LR scheduler
        scheduler.step()
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1 (Macro): {val_f1:.4f} | "
              f"Val AUC (Macro): {val_auc:.4f} | "
              f"Time: {train_time:.1f}s")
              
        # Print class F1s for classes that have positive instances
        f1_strs = []
        for cls_name, f1 in class_f1s.items():
            if f1 > 0:
                f1_strs.append(f"{cls_name}: {f1:.3f}")
        if f1_strs:
            print("  Active Class F1s: " + ", ".join(f1_strs))
            
        # Save best model checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
            print(f"  --> Saved new best checkpoint to {args.save_path} (F1: {val_f1:.4f})")
            
    print(f"\nTraining completed! Best Epoch: {best_epoch} with Val F1 (Macro): {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
