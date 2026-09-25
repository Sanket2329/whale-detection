import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) PyTorch autograd function.
    In the forward pass, it behaves as identity.
    In the backward pass, it negates the incoming gradients and scales by alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    """
    GRL module wrapper.
    """
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)

class ConvBlock(nn.Module):
    """
    Convolutional block with Batch Normalization, ReLU activation, optional Dropout,
    residual skip connection, and max-pooling only along the frequency axis.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Shortcut downsamples along frequency axis using MaxPool2d(2, 1)
        # maps channels using 1x1 Convolution + BN if channels differ.
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Apply shortcut path
        shortcut_x = self.shortcut(x)
        shortcut_x = F.max_pool2d(shortcut_x, kernel_size=(2, 1))
        
        # Apply main path pooling
        out = self.pool(out)
        
        return out + shortcut_x

class DomainClassifier(nn.Module):
    """
    Classifies features into Source (0) or Target (1) domains.
    Input dimension is flat_features (e.g., 256 * 4 = 1024).
    """
    def __init__(self, input_dim=1024, hidden_dim=256, dropout=0.3):
        super(DomainClassifier, self).__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # returns probability of being Target (1)
        )

    def forward(self, x, alpha=1.0):
        # Update GRL coefficient dynamically
        self.grl.alpha = alpha
        x_grl = self.grl(x)
        return self.net(x_grl)

class WhaleSEDModel(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for Sound Event Detection
    equipped with Residual ConvBlocks and a Domain Classifier branch for Unsupervised Domain Adaptation.
    """
    def __init__(self, num_classes=8, base_channels=16, rnn_hidden=128, dropout=0.1):
        super(WhaleSEDModel, self).__init__()
        
        # 2D Residual CNN Frontend
        self.conv1 = ConvBlock(1, base_channels, dropout=dropout)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.conv5 = ConvBlock(base_channels * 8, base_channels * 16, dropout=dropout)
        
        # After 5 max-poolings along frequency axis (size 129):
        # 129 -> 64 -> 32 -> 16 -> 8 -> 4 frequency bins.
        self.flat_features = base_channels * 16 * 4
        
        # Recurrent Sequence Processor
        self.gru = nn.GRU(
            input_size=self.flat_features,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Frame-level Classifier
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        
        # Domain Classifier for Adversarial Alignment
        self.domain_classifier = DomainClassifier(input_dim=self.flat_features, hidden_dim=256, dropout=0.3)

    def forward(self, x, alpha=1.0, return_domain=False):
        # x shape: (B, 1, F, T) -> e.g. (B, 1, 129, 118)
        B, C, F, T = x.shape
        
        # CNN forward pass
        x_feat = self.conv1(x)
        x_feat = self.conv2(x_feat)
        x_feat = self.conv3(x_feat)
        x_feat = self.conv4(x_feat)
        x_feat = self.conv5(x_feat) # Shape: (B, channels, 4, T)
        
        # Extract features for domain classifier if requested
        domain_preds = None
        if return_domain:
            # Pool features over time dimension to get a single vector per clip
            feat_domain = torch.mean(x_feat, dim=3) # Shape: (B, channels, 4)
            feat_domain = feat_domain.reshape(B, -1) # Shape: (B, flat_features)
            domain_preds = self.domain_classifier(feat_domain, alpha=alpha)
        
        # Permute and reshape for RNN
        x = x_feat.permute(0, 3, 1, 2) # Shape: (B, T, channels, 4)
        x = x.reshape(B, T, -1)     # Shape: (B, T, flat_features)
        
        # RNN forward pass
        x, _ = self.gru(x)          # Shape: (B, T, rnn_hidden * 2)
        
        # Classifier output
        logits = self.fc(x)         # Shape: (B, T, num_classes)
        probs = torch.sigmoid(logits)
        
        if return_domain:
            return probs, domain_preds
        return probs

if __name__ == '__main__':
    # Simple self-test
    model = WhaleSEDModel()
    dummy_input = torch.randn(2, 1, 129, 118)
    probs, domains = model(dummy_input, alpha=1.0, return_domain=True)
    print("Dummy input shape:", dummy_input.shape)
    print("Dummy probs shape:", probs.shape)
    print("Dummy domains shape:", domains.shape)
    print("Model parameter count:", sum(p.numel() for p in model.parameters()))
