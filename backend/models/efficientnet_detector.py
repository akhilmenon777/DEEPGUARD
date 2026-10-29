"""
EfficientNet-B4 based Deepfake Detector
Pre-trained on ImageNet and fine-tunable for deepfake detection
"""
import torch
import torch.nn as nn
import timm


class EfficientNetDeepfakeDetector(nn.Module):
    """
    EfficientNet-B4 based binary classifier for deepfake detection.
    Uses transfer learning from ImageNet pre-trained weights.
    """
    
    def __init__(self, model_name='efficientnet_b4', pretrained=True, num_classes=2):
        super().__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get the number of features from backbone
        self.num_features = self.backbone.num_features
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, 3, 224, 224)
        Returns:
            logits: Raw classification scores (batch, 2)
            features: Feature maps before classification (batch, num_features)
        """
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, pooled
    
    def predict_proba(self, x):
        """
        Get prediction probabilities
        Args:
            x: Input tensor
        Returns:
            probabilities: Softmax probabilities (batch, 2)
        """
        logits, _ = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs


def create_efficientnet_detector(pretrained=True, device='cpu'):
    """
    Factory function to create EfficientNet detector
    
    Args:
        pretrained: Whether to use ImageNet pre-trained weights
        device: Device to load model on
        
    Returns:
        model: EfficientNetDeepfakeDetector instance
    """
    model = EfficientNetDeepfakeDetector(
        model_name='efficientnet_b4',
        pretrained=pretrained,
        num_classes=2
    )
    model = model.to(device)
    model.eval()
    return model
