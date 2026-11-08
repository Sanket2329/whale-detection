import torch
import torch.nn as nn
from src.models.dann import DomainClassifier

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Max pooling only on frequency axis (dim=2), keeping temporal length intact
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class WhaleCRNN(nn.Module):
    """Convolutional Recurrent Neural Network for Baleen Whale SED.

    Extracts spatial-temporal features via 5 Conv blocks (frequency pooling only),
    processes time sequence with a 2-layer Bidirectional GRU, and returns raw logits.
    """
    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        
        # 129 freq bins -> 64 -> 32 -> 16 -> 8 -> 4 bins after 5 MaxPool2d((2, 1))
        # 256 channels * 4 bins = 1024 features per time frame
        self.feature_dim = 256 * 4
        
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 2 * hidden_size due to bidirectionality
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 1, Freq=129, Time=118)
        feat = self.cnn(x) # (B, 256, 4, 118)
        B, C, F, T = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, T, C * F) # (B, T, 1024)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        out, _ = self.gru(feat)
        logits = self.classifier(out) # (B, T, num_classes)
        return logits

class WhaleCRNNDANN(WhaleCRNN):
    """CRNN augmented with a Domain Classifier branch via Gradient Reversal."""
    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.domain_classifier = DomainClassifier(in_features=self.feature_dim, hidden_dim=256, dropout=dropout)

    def forward(self, x: torch.Tensor, alpha: float = 1.0, return_domain: bool = False):
        feat = self.forward_features(x)
        out, _ = self.gru(feat)
        logits = self.classifier(out)
        
        if return_domain:
            domain_logits = self.domain_classifier(feat, alpha=alpha)
            return logits, domain_logits
        return logits

