import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label Focal Loss to address extreme class imbalance at the frame level.
    Expects inputs to be predicted probabilities (after sigmoid activation) in [0, 1].
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Clip probabilities to prevent log(0) or log(1) numerical instability
        eps = 1e-6
        inputs = torch.clamp(inputs, min=eps, max=1.0 - eps)
        
        # Calculate BCE loss per element
        bce = - (targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))
        
        # Modulating factor (1 - pt)^gamma
        p_t = inputs * targets + (1.0 - inputs) * (1.0 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        loss = modulating_factor * bce
        
        # Apply class imbalance scaling alpha
        if self.alpha is not None and self.alpha >= 0:
            alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            loss = alpha_factor * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
