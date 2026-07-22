import torch
import torch.nn as nn
import torch.nn.functional as F

class WhaleSEDModel(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for Sound Event Detection.
    Keeps the time dimension constant through Conv2D blocks by only pooling along the frequency axis.
    """
    def __init__(self, num_classes=8, base_channels=16, rnn_hidden=128):
        super(WhaleSEDModel, self).__init__()
        
        # 2D CNN Frontend
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) # frequency pooled by 2, time untouched
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) # frequency pooled by 2, time untouched
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) # frequency pooled by 2, time untouched
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) # frequency pooled by 2, time untouched
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) # frequency pooled by 2, time untouched
        )
        
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

    def forward(self, x):
        # x shape: (B, 1, F, T) -> e.g. (B, 1, 129, 118)
        B, C, F, T = x.shape
        
        # CNN forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) # Shape: (B, channels, 4, T)
        
        # Permute and reshape for RNN
        x = x.permute(0, 3, 1, 2) # Shape: (B, T, channels, 4)
        x = x.reshape(B, T, -1)     # Shape: (B, T, flat_features)
        
        # RNN forward pass
        x, _ = self.gru(x)          # Shape: (B, T, rnn_hidden * 2)
        
        # Classifier output
        logits = self.fc(x)         # Shape: (B, T, num_classes)
        probs = torch.sigmoid(logits)
        
        return probs

if __name__ == '__main__':
    # Simple self-test
    model = WhaleSEDModel()
    dummy_input = torch.randn(2, 1, 129, 118)
    dummy_output = model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Dummy output shape:", dummy_output.shape)
    print("Model parameter count:", sum(p.numel() for p in model.parameters()))
