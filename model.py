import torch
import torch.nn as nn


class ForestPersonCNN(nn.Module):
    """
    Simple CNN binary classifier for detecting humans in forest images.

    Architecture:
        3 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
        with increasing filter sizes (32 -> 64 -> 128), followed by
        two fully connected layers and a final output layer with 2 units.

    Input : (batch, 3, 224, 224)
    Output: (batch, 2)  -- raw logits for [no_human, human]
    """

    def __init__(self, dropout=0.5):
        super(ForestPersonCNN, self).__init__()

        # --- Convolutional blocks ---
        # Block 1: 3 -> 32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 224 -> 112
        )

        # Block 2: 32 -> 64 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 112 -> 56
        )

        # Block 3: 64 -> 128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 56 -> 28
        )

        # --- Classifier head ---
        # After 3 MaxPool(2x2): 224 / 8 = 28
        # Feature map size: 128 * 28 * 28 = 100352
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),                       # 2 output classes
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------------
# Quick sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = ForestPersonCNN()
    print(model)
    print()

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Test forward pass with a dummy batch
    dummy = torch.randn(4, 3, 224, 224)   # batch of 4
    output = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {output.shape}  (expected: [4, 2])")
    print("\nmodel.py looks good!")
