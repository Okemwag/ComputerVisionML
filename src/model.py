"""
Medical Image Analysis - Model Architectures

U-Net and DeepLabV3+ implementations for medical image segmentation and classification.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation and classification.
    
    Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 2, 
        depth: int = 4, 
        start_filters: int = 64,
        bilinear: bool = True,
        dropout: float = 0.2,
        task_type: str = 'classification'
    ):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels/classes
            depth: Depth of the U-Net (number of down/up blocks)
            start_filters: Number of filters in first layer
            bilinear: Use bilinear upsampling instead of transpose convolution
            dropout: Dropout rate
            task_type: 'classification' or 'segmentation'
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.start_filters = start_filters
        self.bilinear = bilinear
        self.dropout = dropout
        self.task_type = task_type
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, start_filters, dropout)
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        in_ch = start_filters
        for i in range(depth):
            out_ch = start_filters * (2 ** (i + 1))
            self.encoder.append(Down(in_ch, out_ch, dropout))
            in_ch = out_ch
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in range(depth):
            out_ch = start_filters * (2 ** (depth - i - 1))
            self.decoder.append(Up(in_ch, out_ch, bilinear, dropout))
            in_ch = out_ch
        
        # Output layers
        if task_type == 'segmentation':
            self.outc = OutConv(start_filters, out_channels)
        elif task_type == 'classification':
            # Global average pooling + classifier
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(start_filters, start_filters // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(start_filters // 2, out_channels)
            )
        else:  # Both tasks
            self.outc = OutConv(start_filters, out_channels)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(start_filters, start_filters // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(start_filters // 2, out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net."""
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.inc(x)
        skip_connections.append(x)
        
        # Encoder path
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (bottom of U)
        skip_connections.pop()
        
        # Decoder path
        for i, up in enumerate(self.decoder):
            skip = skip_connections.pop()
            x = up(x, skip)
        
        # Output based on task type
        if self.task_type == 'segmentation':
            return self.outc(x)
        elif self.task_type == 'classification':
            # Global average pooling
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)
        else:  # Both tasks
            # Segmentation output
            seg_out = self.outc(x)
            
            # Classification output
            cls_features = self.global_pool(x)
            cls_features = torch.flatten(cls_features, 1)
            cls_out = self.classifier(cls_features)
            
            return {'segmentation': seg_out, 'classification': cls_out}
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract feature maps at different scales."""
        features = []
        
        # Initial convolution
        x = self.inc(x)
        features.append(x)
        
        # Encoder path
        for down in self.encoder:
            x = down(x)
            features.append(x)
        
        return features
    
    def freeze_encoder(self):
        """Freeze encoder weights for transfer learning."""
        self.inc.requires_grad_(False)
        for down in self.encoder:
            down.requires_grad_(False)
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        self.inc.requires_grad_(True)
        for down in self.encoder:
            down.requires_grad_(True)


class ClassificationHead(nn.Module):
    """Classification head for contrast detection."""
    
    def __init__(self, in_features: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SegmentationHead(nn.Module):
    """Segmentation head for pixel-level predictions."""
    
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.segmentation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmentation(x)


def create_unet_model(config: dict) -> UNet:
    """
    Factory function to create U-Net model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        UNet model instance
    """
    model_config = config.get('model', {})
    
    return UNet(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('num_classes', 2),
        depth=model_config.get('depth', 4),
        start_filters=model_config.get('start_filters', 64),
        bilinear=model_config.get('bilinear', True),
        dropout=model_config.get('dropout', 0.2),
        task_type=model_config.get('task_type', 'classification')
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = (1, 1, 256, 256)) -> dict:
    """
    Generate model summary with parameter counts and output shapes.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        
    Returns:
        Dictionary with model information
    """
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Test forward pass
    dummy_input = torch.randn(input_size)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        if isinstance(output, dict):
            output_shapes = {k: v.shape for k, v in output.items()}
        else:
            output_shapes = output.shape
    except Exception as e:
        output_shapes = f"Error: {str(e)}"
    
    summary = {
        'model_name': model.__class__.__name__,
        'input_size': input_size,
        'output_shapes': output_shapes,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
    
    return summary


class ASPPConv(nn.Module):
    """Atrous Spatial Pyramid Pooling convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPPPooling(nn.Module):
    """ASPP pooling block with global average pooling."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.gap(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        dilations = [1, 6, 12, 18]
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        for dilation in dilations[1:]:
            self.convs.append(ASPPConv(in_channels, out_channels, dilation))
        
        self.convs.append(ASPPPooling(in_channels, out_channels))
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3PlusBackbone(nn.Module):
    """Simplified ResNet-like backbone for DeepLabV3+."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Layer 1 (low-level features)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Layer 4 (high-level features with atrous convolution)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        x = self.conv1(x)
        
        low_level = self.layer1(x)  # Low-level features for skip connection
        
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)  # High-level features for ASPP
        
        return low_level, high_level


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for medical image segmentation and classification.
    
    Paper: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        num_classes: int = 2, 
        backbone: str = 'custom',
        task_type: str = 'classification'
    ):
        """
        Initialize DeepLabV3+ model.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            backbone: Backbone architecture ('custom' for simplified version)
            task_type: 'classification', 'segmentation', or 'both'
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.task_type = task_type
        
        # Backbone
        self.backbone = DeepLabV3PlusBackbone(in_channels)
        
        # ASPP module
        self.aspp = ASPP(512, 256)
        
        # Decoder
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        if task_type == 'segmentation':
            self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)
        elif task_type == 'classification':
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        else:  # Both tasks
            self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLabV3+."""
        input_size = x.shape[-2:]
        
        # Backbone
        low_level_features, high_level_features = self.backbone(x)
        
        # ASPP
        aspp_features = self.aspp(high_level_features)
        
        # Upsample ASPP features
        aspp_features = F.interpolate(
            aspp_features, size=low_level_features.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        # Process low-level features
        low_level_features = self.low_level_conv(low_level_features)
        
        # Concatenate features
        features = torch.cat([aspp_features, low_level_features], dim=1)
        
        # Decoder
        features = self.decoder(features)
        
        # Task-specific outputs
        if self.task_type == 'segmentation':
            output = self.segmentation_head(features)
            return F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        elif self.task_type == 'classification':
            return self.classification_head(features)
        else:  # Both tasks
            seg_output = self.segmentation_head(features)
            seg_output = F.interpolate(seg_output, size=input_size, mode='bilinear', align_corners=False)
            cls_output = self.classification_head(features)
            
            return {'segmentation': seg_output, 'classification': cls_output}
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the decoder for analysis."""
        low_level_features, high_level_features = self.backbone(x)
        aspp_features = self.aspp(high_level_features)
        
        aspp_features = F.interpolate(
            aspp_features, size=low_level_features.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        low_level_features = self.low_level_conv(low_level_features)
        features = torch.cat([aspp_features, low_level_features], dim=1)
        features = self.decoder(features)
        
        return features
    
    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        self.backbone.requires_grad_(False)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        self.backbone.requires_grad_(True)


def create_deeplabv3_model(config: dict) -> DeepLabV3Plus:
    """
    Factory function to create DeepLabV3+ model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DeepLabV3Plus model instance
    """
    model_config = config.get('model', {})
    
    return DeepLabV3Plus(
        in_channels=model_config.get('in_channels', 1),
        num_classes=model_config.get('num_classes', 2),
        backbone=model_config.get('backbone', 'custom'),
        task_type=model_config.get('task_type', 'classification')
    )


def create_model(config: dict) -> nn.Module:
    """
    Factory function to create any model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config.get('model', {})
    architecture = model_config.get('architecture', 'unet').lower()
    
    if architecture == 'unet':
        return create_unet_model(config)
    elif architecture in ['deeplabv3', 'deeplabv3plus', 'deeplabv3+']:
        return create_deeplabv3_model(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def compare_models(input_size: tuple = (1, 1, 256, 256)) -> dict:
    """
    Compare U-Net and DeepLabV3+ models.
    
    Args:
        input_size: Input tensor size for comparison
        
    Returns:
        Dictionary with comparison results
    """
    models = {
        'UNet': UNet(in_channels=1, out_channels=2, depth=4, start_filters=64),
        'DeepLabV3+': DeepLabV3Plus(in_channels=1, num_classes=2)
    }
    
    comparison = {}
    
    for name, model in models.items():
        summary = model_summary(model, input_size)
        comparison[name] = {
            'parameters': summary['total_parameters'],
            'size_mb': summary['model_size_mb'],
            'output_shape': summary['output_shapes']
        }
    
    return comparison