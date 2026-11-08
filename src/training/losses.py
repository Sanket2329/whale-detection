import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Multi-label Focal Loss operating on raw logits.

    Down-weights easy background frames and focuses gradient updates on hard call events.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        
        # p_t: probability of true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        
        # alpha_t: class weighting
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        
        # Modulating factor
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-8), self.gamma)
        loss = focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

