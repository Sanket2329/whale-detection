import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

class DomainClassifier(nn.Module):
    """Domain Classifier head for DANN.

    Distinguishes whether feature representations originate from
    the source site (label 0) or target site (label 1).
    """
    def __init__(self, in_features: int = 1024, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # features shape: (Batch, Time, Channels) or (Batch, In_Features)
        if features.dim() == 3:
            # Global average pool over time dimension
            features = features.mean(dim=1)
        reversed_features = grad_reverse(features, alpha)
        return self.network(reversed_features)

