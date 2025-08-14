# src/model.py
import torch.nn as nn
from torchvision import models
import torch.nn.init as init
from . import config

class DeepfakeDetector(nn.Module):
    """
    VGG19-based deepfake detector with a reduced capacity head and progressive unfreezing.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg19.features
        
        # Classifier head with reduced capacity to fight overfitting
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 1)
        )
        self._initialize_weights()
        self.freeze_backbone()

    def _initialize_weights(self):
        """Initializes the weights of the classifier head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freezes all parameters in the VGG19 backbone."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_last_n_modules(self, n):
        """
        Unfreezes the last `n` modules of the VGG19 features for fine-tuning.
        """
        if n <= 0: 
            return
        children = list(self.features.children())
        for module in children[-n:]:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)