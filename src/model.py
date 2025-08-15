# src/model.py
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init
from . import config

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module as described in the paper.
    It learns a mask to emphasize important spatial features and adds it
    back to the original feature map via a residual connection.
    Output: O = I + (T * M)
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # Pre-activation of the input tensor I
        self.pre_process = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Trunk branch to process features (T)
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
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        normalized_x = self.pre_process(x)
        trunk_features = self.trunk_branch(normalized_x)
        attention_mask = self.mask_branch(normalized_x)
        # Apply attention: O = I + (T * M)
        return identity + (trunk_features * attention_mask)

class DeepfakeDetector(nn.Module):
    """The complete Deepfake Detector model, combining a VGG19 backbone with Spatial Attention."""
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        # Load VGG19 with pretrained weights
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg19.features
        
        # VGG19 produces 512 output channels before the classifier
        vgg_output_channels = 512
        self.attention = SpatialAttention(in_channels=vgg_output_channels)
        
        # New classifier head to replace the original VGG classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(vgg_output_channels * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 1) # Single output for binary classification
        )
        
        # Initialize weights for the new classifier
        self._initialize_weights()

        # Freeze backbone by default for Stage 1 training
        if pretrained:
            self.freeze_backbone()

    def _initialize_weights(self):
        """Initializes weights of the classifier layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freezes the convolutional base of VGG19."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the convolutional base for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x.squeeze(1) # Remove last dimension for loss calculation