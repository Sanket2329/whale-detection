import os
import torch
import subprocess
from typing import Any, Dict

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        return "unknown"

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    seed: int,
    class_thresholds: Dict[str, float] = None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "seed": seed,
        "git_commit": get_git_commit_hash(),
        "class_thresholds": class_thresholds or {},
        "pytorch_version": torch.__version__
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str, device: str = "cpu", weights_only: bool = False) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=weights_only)

