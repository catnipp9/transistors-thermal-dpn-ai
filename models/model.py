"""
Model architectures for DPN classification from plantar thermograms
Includes YOLOv11 (primary), CNN, ResNet, and classical ML approaches
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


# ============== YOLOv11 Classification Model ==============

class YOLOv11DPNClassifier:
    """
    YOLOv11-based classifier for DPN detection from plantar thermograms.

    Uses Ultralytics YOLOv11 in classification mode (yolo11n-cls / yolo11s-cls),
    which provides higher accuracy than the custom CNN especially at low sample sizes
    thanks to pre-trained ImageNet weights and YOLO's advanced training pipeline.

    Model size variants (tradeoff between speed and accuracy):
        - 'yolo11n-cls': nano   – fastest, lowest accuracy
        - 'yolo11s-cls': small  – recommended default
        - 'yolo11m-cls': medium – more accurate, slower
        - 'yolo11l-cls': large
        - 'yolo11x-cls': extra-large – most accurate, slowest

    Usage:
        model = YOLOv11DPNClassifier(variant='yolo11s-cls')
        model.train(data_yaml='path/to/dataset.yaml', epochs=50)
        results = model.predict('path/to/image.png')
    """

    # Map friendly class names to YOLO label indices
    CLASS_NAMES = {0: "Control", 1: "Diabetic"}
    CLASS_INDICES = {"Control": 0, "Diabetic": 1}

    def __init__(
        self,
        variant: str = 'yolo11m-cls',
        pretrained_path: Optional[str] = None,
        num_classes: int = 2,
        device: Optional[str] = None
    ):
        """
        Args:
            variant: YOLO model variant (e.g. 'yolo11s-cls'). Ignored when
                     pretrained_path points to a fine-tuned .pt file.
            pretrained_path: Path to a previously fine-tuned weights file.
                             If None, downloads / uses cached Ultralytics weights.
            num_classes: Number of output classes (2 for binary DPN detection).
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLOv11. "
                "Install it with: pip install ultralytics>=8.3.0"
            ) from e

        self.variant = variant
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model: fine-tuned weights take priority, else use variant weights
        weights = pretrained_path if pretrained_path else variant
        self.model = YOLO(weights)

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 224,
        batch: int = 16,
        lr0: float = 0.001,
        patience: int = 10,
        save_dir: str = "checkpoints",
        **kwargs
    ) -> object:
        """
        Fine-tune YOLOv11 on the DPN thermogram dataset.

        Args:
            data_yaml: Path to the YOLO dataset directory OR dataset.yaml file.
                       For classification mode Ultralytics requires the directory,
                       so if a .yaml file path is passed the parent directory is used.
            epochs: Maximum training epochs.
            imgsz: Image size fed to the network (images are resized to imgsz×imgsz).
            batch: Batch size.
            lr0: Initial learning rate.
            patience: Early-stopping patience (epochs without improvement).
            save_dir: Directory where training results and best weights are saved.
            **kwargs: Additional keyword arguments forwarded to YOLO.train().

        Returns:
            Ultralytics Results object from training.
        """
        from pathlib import Path as _Path

        # Ultralytics classify mode needs the dataset directory, not a YAML file
        data_path = _Path(data_yaml)
        if data_path.suffix in {".yaml", ".yml"}:
            data_path = data_path.parent

        # Augmentation defaults tuned for small thermal foot images.
        # Any can be overridden by the caller via **kwargs.
        augmentation_defaults = dict(
            # Geometric
            fliplr=0.5,       # horizontal flip — left/right foot mirror symmetry
            flipud=0.0,       # no vertical flip — foot orientation matters
            degrees=15,       # rotation ±15° for camera tilt variation
            translate=0.1,    # translate ±10%
            scale=0.2,        # scale ±20%
            shear=5.0,        # shear ±5°
            # Colour / intensity (thermal colourmap variation)
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            # Regularisation
            erasing=0.3,      # random erasing — simulates partial occlusion
            mixup=0.1,        # mixup helps generalise on small datasets
            # LR schedule — cosine annealing prevents getting stuck in local minima
            cos_lr=True,
            # Optimizer
            weight_decay=0.0005,
            warmup_epochs=5,   # longer warmup stabilises early training
        )
        augmentation_defaults.update(kwargs)

        results = self.model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            patience=patience,
            project=save_dir,
            name="yolo11_dpn",
            device=self.device,
            exist_ok=True,
            **augmentation_defaults
        )
        return results

    def predict(self, image) -> dict:
        """
        Classify a single thermal image.

        Args:
            image: File path (str/Path), PIL Image, or numpy array (H×W×3 uint8).
            conf: Confidence threshold (not strictly needed for classification,
                  included for API consistency).

        Returns:
            Dict with keys: prediction, class_index, confidence, is_diabetic,
            probabilities ({"Control": float, "Diabetic": float}).
        """
        import numpy as np
        from PIL import Image as PILImage

        # Normalise input to PIL RGB
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image.astype("uint8")).convert("RGB")
        elif not isinstance(image, PILImage.Image):
            image = PILImage.open(image).convert("RGB")

        results = self.model(image, verbose=False)
        result = results[0]

        # probs is a Probs object from ultralytics
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        prob_array = probs.data.cpu().numpy().tolist()

        # Guard: handle cases where model was trained with only 1 output class
        control_prob = prob_array[0] * 100 if len(prob_array) > 0 else 0.0
        diabetic_prob = prob_array[1] * 100 if len(prob_array) > 1 else 0.0

        return {
            "prediction": self.CLASS_NAMES.get(top1_idx, str(top1_idx)),
            "class_index": top1_idx,
            "confidence": round(top1_conf * 100, 2),
            "is_diabetic": top1_idx == 1,
            "probabilities": {
                "Control": round(control_prob, 2),
                "Diabetic": round(diabetic_prob, 2),
            }
        }

    def save(self, path: str):
        """Export the current model weights to a .pt file."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "YOLOv11DPNClassifier":
        """Load a previously saved YOLOv11 DPN model from a .pt file."""
        instance = cls.__new__(cls)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLOv11. "
                "Install it with: pip install ultralytics>=8.3.0"
            ) from e

        instance.variant = path
        instance.num_classes = 2
        instance.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        instance.CLASS_NAMES = cls.CLASS_NAMES
        instance.CLASS_INDICES = cls.CLASS_INDICES
        instance.model = YOLO(path)
        return instance


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
        model_name: Name of the model. Use 'yolo11' (recommended) for best accuracy.
        num_classes: Number of output classes
        input_channels: Number of input channels (3 for RGB, 1 for temperature)

    Returns:
        Model instance
    """
    # YOLOv11 variants — constructed on demand to avoid loading weights at import time
    yolo_variants = {
        'yolo11':   'yolo11m-cls',   # default (medium) – better accuracy for imbalanced data
        'yolo11n':  'yolo11n-cls',   # nano – fastest
        'yolo11s':  'yolo11s-cls',   # small
        'yolo11m':  'yolo11m-cls',   # medium
        'yolo11l':  'yolo11l-cls',   # large
        'yolo11x':  'yolo11x-cls',   # extra-large – most accurate
    }

    if model_name in yolo_variants:
        variant = kwargs.pop('variant', yolo_variants[model_name])
        pretrained_path = kwargs.pop('pretrained_path', None)
        return YOLOv11DPNClassifier(
            variant=variant,
            pretrained_path=pretrained_path,
            num_classes=num_classes,
        )

    models = {
        # Deep Learning (CNN-based)
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
        raise ValueError(f"Unknown model: {model_name}. Available: {list(yolo_variants.keys()) + list(models.keys())}")

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
