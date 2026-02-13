"""
Model architectures for DPN classification from plantar thermograms
Includes both deep learning (CNN) and classical ML approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ThermogramCNN(nn.Module):
    """
    CNN architecture for plantar thermogram classification.
    Designed for input size (3, 168, 65) - RGB thermogram images.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        input_channels: int = 3
    ):
        super(ThermogramCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 168x65 -> 84x32
            nn.Dropout2d(0.25),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 84x32 -> 42x16
            nn.Dropout2d(0.25),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 42x16 -> 21x8
            nn.Dropout2d(0.25),

            # Block 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for faster training and inference.
    Good for limited data scenarios.
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3
    ):
        super(LightweightCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class TemperatureMatrixCNN(nn.Module):
    """
    CNN for single-channel temperature matrix input.
    Input: (1, 168, 65) temperature values
    """

    def __init__(self, num_classes: int = 2):
        super(TemperatureMatrixCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ThermogramResNet(nn.Module):
    """
    ResNet-style architecture for thermogram classification.
    Better for deeper feature learning.
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3
    ):
        super(ThermogramResNet, self).__init__()

        self.in_channels = 32

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============== Classical ML Models ==============

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_random_forest(n_estimators: int = 100, random_state: int = 42):
    """Create Random Forest classifier with preprocessing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ))
    ])


def create_svm(kernel: str = 'rbf', C: float = 1.0, random_state: int = 42):
    """Create SVM classifier with preprocessing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            kernel=kernel,
            C=C,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        ))
    ])


def create_gradient_boosting(n_estimators: int = 100, random_state: int = 42):
    """Create Gradient Boosting classifier with preprocessing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=random_state
        ))
    ])


def create_mlp(hidden_layers: Tuple[int, ...] = (256, 128, 64), random_state: int = 42):
    """Create MLP classifier with preprocessing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state
        ))
    ])


def create_logistic_regression(random_state: int = 42):
    """Create Logistic Regression classifier with preprocessing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        ))
    ])


def get_model(
    model_name: str,
    num_classes: int = 2,
    input_channels: int = 3,
    **kwargs
):
    """
    Factory function to get model by name.

    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        input_channels: Number of input channels (3 for RGB, 1 for temperature)

    Returns:
        Model instance
    """
    models = {
        # Deep Learning
        'cnn': ThermogramCNN(num_classes, input_channels=input_channels),
        'lightweight_cnn': LightweightCNN(num_classes, input_channels=input_channels),
        'temp_cnn': TemperatureMatrixCNN(num_classes),
        'resnet': ThermogramResNet(num_classes, input_channels=input_channels),

        # Classical ML
        'random_forest': create_random_forest(**kwargs),
        'svm': create_svm(**kwargs),
        'gradient_boosting': create_gradient_boosting(**kwargs),
        'mlp': create_mlp(**kwargs),
        'logistic_regression': create_logistic_regression(**kwargs),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name]


if __name__ == "__main__":
    # Test model architectures
    print("Testing model architectures...")

    # Test CNN
    model = ThermogramCNN(num_classes=2)
    x = torch.randn(4, 3, 168, 65)  # Batch of 4 RGB images
    out = model(x)
    print(f"ThermogramCNN output shape: {out.shape}")

    # Test Temperature CNN
    model = TemperatureMatrixCNN(num_classes=2)
    x = torch.randn(4, 1, 168, 65)  # Batch of 4 temperature matrices
    out = model(x)
    print(f"TemperatureMatrixCNN output shape: {out.shape}")

    # Test ResNet
    model = ThermogramResNet(num_classes=2)
    x = torch.randn(4, 3, 168, 65)
    out = model(x)
    print(f"ThermogramResNet output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ThermogramResNet total parameters: {total_params:,}")
