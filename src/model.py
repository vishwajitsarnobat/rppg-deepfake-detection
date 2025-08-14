# src/model.py

import torch.nn as nn
from torchvision import models

from . import config

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module as depicted in Figure 6 of the paper.
    It learns a mask to emphasize important spatial features and adds it
    back to the original feature map.
    Output: O = I + (T * M)
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # Initial normalization of the input tensor I
        self.pre_process = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Backbone branch to process features (T)
        self.trunk_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Soft mask branch to generate the attention map (M)
        self.mask_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False), # Output a single-channel mask
            nn.Sigmoid() # Scale mask values between 0 and 1
        )

    def forward(self, x):
        identity = x
        normalized_x = self.pre_process(x)

        # Calculate trunk features (T)
        trunk_features = self.trunk_branch(normalized_x)

        # Calculate attention mask (M)
        attention_mask = self.mask_branch(normalized_x)

        # Apply attention: O = I + (T * M) as per the paper's formula
        # Broadcasting handles the multiplication of (C, H, W) features by (1, H, W) mask
        output = identity + (trunk_features * attention_mask)

        return output


class DeepfakeDetector(nn.Module):
    """
    The complete Deepfake Detector model, combining a VGG19 backbone with the
    Spatial Attention module and a custom classifier head.
    """
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()

        # 1. Load VGG19 and use its feature extractor part
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg19.features

        # --- FIX: Freeze the backbone initially for transfer learning ---
        # It can be unfrozen later for fine-tuning.
        self.freeze_backbone()

        vgg_output_channels = 512
        # VGG19 with 240x240 input gives 7x7 feature map
        vgg_output_spatial_dim = 7

        # 2. Add the Spatial Attention module (this will be trainable)
        self.attention = SpatialAttention(in_channels=vgg_output_channels)

        # 3. Define the final classifier head (this will be trainable)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(vgg_output_channels * vgg_output_spatial_dim**2, 1024),
            nn.ReLU(True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 1) # Final output for binary classification
        )

    def freeze_backbone(self):
        """Freezes the gradients for all layers in the VGG19 feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False
        # Ensure the feature extractor is in eval mode if frozen
        self.features.eval()


    def unfreeze_backbone(self):
        """Unfreezes the backbone for later fine-tuning if needed."""
        for param in self.features.parameters():
            param.requires_grad = True
        self.features.train()

    def forward(self, x):
        # --- FIX: Removed the `torch.no_grad()` context manager. ---
        # The `requires_grad` status of the parameters is sufficient to control
        # whether gradients are computed for the backbone. This allows for
        # proper fine-tuning when the backbone is unfrozen.
        x = self.features(x)

        # The attention and classifier layers are always trainable
        x = self.attention(x)
        x = self.classifier(x)
        return x