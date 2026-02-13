"""
Inference utilities for DPN classification models
Handles model loading and prediction
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import joblib
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import get_model
from models.data_loader import get_transforms


class DPNClassifier:
    """
    Unified classifier for DPN detection.
    Supports both CNN (PyTorch) and classical ML (sklearn) models.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "cnn",  # "cnn" or "sklearn"
        device: str = None
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to saved model file (.pth for CNN, .joblib for sklearn)
            model_type: Type of model - "cnn" or "sklearn"
            device: Device for CNN inference ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.transform = None

        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if self.model_type == "cnn":
            # Load PyTorch CNN model
            self.model = get_model("lightweight_cnn", num_classes=2, input_channels=3)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # Setup transform for inference
            self.transform = get_transforms(train=False)

            print(f"Loaded CNN model from {self.model_path}")
            print(f"Running on: {self.device}")

        elif self.model_type == "sklearn":
            # Load sklearn model
            self.model = joblib.load(self.model_path)
            print(f"Loaded sklearn model from {self.model_path}")

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for CNN inference.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            Preprocessed tensor ready for model
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

        # Apply transforms
        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def preprocess_csv(self, csv_path: Union[str, Path], target_shape: Tuple[int, int] = (168, 65)) -> np.ndarray:
        """
        Preprocess a CSV temperature matrix for sklearn inference.

        Args:
            csv_path: Path to CSV file
            target_shape: Target shape for resizing

        Returns:
            Flattened feature vector
        """
        import pandas as pd
        from scipy.ndimage import zoom

        data = pd.read_csv(csv_path, header=None).values.astype(np.float32)

        # Resize if needed
        if data.shape != target_shape:
            zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
            data = zoom(data, zoom_factors, order=1)

        return data.flatten().reshape(1, -1)

    def predict(
        self,
        input_data: Union[str, Path, Image.Image, np.ndarray],
        return_proba: bool = True
    ) -> Dict:
        """
        Make a prediction on input data.

        Args:
            input_data: Image path, PIL Image, numpy array, or CSV path
            return_proba: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        if self.model_type == "cnn":
            return self._predict_cnn(input_data, return_proba)
        else:
            return self._predict_sklearn(input_data, return_proba)

    def _predict_cnn(self, image: Union[str, Path, Image.Image, np.ndarray], return_proba: bool) -> Dict:
        """Make prediction using CNN model."""
        # Preprocess
        tensor = self.preprocess_image(image).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        # Map class index to label
        class_labels = {0: "Control", 1: "Diabetic"}

        result = {
            "prediction": class_labels[predicted_class],
            "class_index": predicted_class,
            "confidence": round(confidence * 100, 2),
            "is_diabetic": predicted_class == 1
        }

        if return_proba:
            result["probabilities"] = {
                "Control": round(probabilities[0, 0].item() * 100, 2),
                "Diabetic": round(probabilities[0, 1].item() * 100, 2)
            }

        return result

    def _predict_sklearn(self, csv_path: Union[str, Path], return_proba: bool) -> Dict:
        """Make prediction using sklearn model."""
        # Preprocess
        features = self.preprocess_csv(csv_path)

        # Inference
        predicted_class = self.model.predict(features)[0]

        # Map class index to label
        class_labels = {0: "Control", 1: "Diabetic"}

        result = {
            "prediction": class_labels[predicted_class],
            "class_index": int(predicted_class),
            "is_diabetic": predicted_class == 1
        }

        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            result["confidence"] = round(max(probabilities) * 100, 2)
            result["probabilities"] = {
                "Control": round(probabilities[0] * 100, 2),
                "Diabetic": round(probabilities[1] * 100, 2)
            }

        return result


def calculate_asymmetry(
    left_temps: np.ndarray,
    right_temps: np.ndarray
) -> Dict:
    """
    Calculate temperature asymmetry between left and right foot.

    Asymmetry is a key clinical indicator for DPN - significant temperature
    differences between feet may indicate nerve damage.

    Args:
        left_temps: Temperature matrix for left foot
        right_temps: Temperature matrix for right foot

    Returns:
        Dictionary with asymmetry metrics
    """
    from scipy.ndimage import zoom

    # Ensure same shape
    target_shape = (168, 65)

    if left_temps.shape != target_shape:
        zoom_factors = (target_shape[0] / left_temps.shape[0], target_shape[1] / left_temps.shape[1])
        left_temps = zoom(left_temps, zoom_factors, order=1)

    if right_temps.shape != target_shape:
        zoom_factors = (target_shape[0] / right_temps.shape[0], target_shape[1] / right_temps.shape[1])
        right_temps = zoom(right_temps, zoom_factors, order=1)

    # Flip right foot horizontally for alignment comparison
    right_flipped = np.fliplr(right_temps)

    # Calculate differences
    diff = np.abs(left_temps - right_flipped)

    # Asymmetry metrics
    mean_asymmetry = float(np.mean(diff))
    max_asymmetry = float(np.max(diff))
    std_asymmetry = float(np.std(diff))

    # Temperature stats per foot
    left_mean = float(np.mean(left_temps))
    right_mean = float(np.mean(right_temps))
    mean_temp_diff = abs(left_mean - right_mean)

    # Clinical threshold: >2.2°C difference is often considered significant
    ASYMMETRY_THRESHOLD = 2.2
    is_significant = mean_temp_diff > ASYMMETRY_THRESHOLD or max_asymmetry > (ASYMMETRY_THRESHOLD * 2)

    return {
        "mean_asymmetry": round(mean_asymmetry, 3),
        "max_asymmetry": round(max_asymmetry, 3),
        "std_asymmetry": round(std_asymmetry, 3),
        "left_foot_mean_temp": round(left_mean, 2),
        "right_foot_mean_temp": round(right_mean, 2),
        "mean_temp_difference": round(mean_temp_diff, 3),
        "asymmetry_significant": is_significant,
        "threshold_used": ASYMMETRY_THRESHOLD
    }


def predict_patient(
    cnn_classifier: Optional[DPNClassifier],
    sklearn_classifier: Optional[DPNClassifier],
    left_image: Optional[Union[str, Path, Image.Image]] = None,
    right_image: Optional[Union[str, Path, Image.Image]] = None,
    left_csv: Optional[Union[str, Path]] = None,
    right_csv: Optional[Union[str, Path]] = None,
    left_temps: Optional[np.ndarray] = None,
    right_temps: Optional[np.ndarray] = None
) -> Dict:
    """
    Make a combined prediction for a patient using both feet.

    This provides a more clinically accurate assessment by:
    1. Analyzing each foot individually
    2. Calculating temperature asymmetry between feet
    3. Combining results for final diagnosis

    Args:
        cnn_classifier: CNN classifier for images
        sklearn_classifier: sklearn classifier for temperature values
        left_image: Left foot thermal image
        right_image: Right foot thermal image
        left_csv: Path to left foot temperature CSV
        right_csv: Path to right foot temperature CSV
        left_temps: Left foot temperature matrix (numpy array)
        right_temps: Right foot temperature matrix (numpy array)

    Returns:
        Combined diagnosis with individual foot results and asymmetry analysis
    """
    import pandas as pd

    results = {
        "left_foot": None,
        "right_foot": None,
        "asymmetry": None,
        "combined_prediction": None,
        "combined_confidence": None,
        "is_diabetic": None,
        "diagnosis_factors": []
    }

    # Predict left foot
    if left_image is not None and cnn_classifier is not None:
        results["left_foot"] = cnn_classifier.predict(left_image, return_proba=True)
        results["left_foot"]["input_type"] = "image"
    elif left_csv is not None and sklearn_classifier is not None:
        results["left_foot"] = sklearn_classifier.predict(left_csv, return_proba=True)
        results["left_foot"]["input_type"] = "csv"
        # Load temps for asymmetry calculation
        if left_temps is None:
            left_temps = pd.read_csv(left_csv, header=None).values.astype(np.float32)
    elif left_temps is not None and sklearn_classifier is not None:
        # Save temp array to temporary CSV for prediction
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(left_temps).to_csv(f.name, header=False, index=False)
            results["left_foot"] = sklearn_classifier.predict(f.name, return_proba=True)
            results["left_foot"]["input_type"] = "temperature_array"
            Path(f.name).unlink(missing_ok=True)

    # Predict right foot
    if right_image is not None and cnn_classifier is not None:
        results["right_foot"] = cnn_classifier.predict(right_image, return_proba=True)
        results["right_foot"]["input_type"] = "image"
    elif right_csv is not None and sklearn_classifier is not None:
        results["right_foot"] = sklearn_classifier.predict(right_csv, return_proba=True)
        results["right_foot"]["input_type"] = "csv"
        # Load temps for asymmetry calculation
        if right_temps is None:
            right_temps = pd.read_csv(right_csv, header=None).values.astype(np.float32)
    elif right_temps is not None and sklearn_classifier is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(right_temps).to_csv(f.name, header=False, index=False)
            results["right_foot"] = sklearn_classifier.predict(f.name, return_proba=True)
            results["right_foot"]["input_type"] = "temperature_array"
            Path(f.name).unlink(missing_ok=True)

    # Calculate asymmetry if we have temperature data for both feet
    if left_temps is not None and right_temps is not None:
        results["asymmetry"] = calculate_asymmetry(left_temps, right_temps)
    elif left_csv is not None and right_csv is not None:
        left_temps = pd.read_csv(left_csv, header=None).values.astype(np.float32)
        right_temps = pd.read_csv(right_csv, header=None).values.astype(np.float32)
        results["asymmetry"] = calculate_asymmetry(left_temps, right_temps)

    # Combined diagnosis logic
    left_diabetic = results["left_foot"]["is_diabetic"] if results["left_foot"] else None
    right_diabetic = results["right_foot"]["is_diabetic"] if results["right_foot"] else None
    left_conf = results["left_foot"].get("confidence", 0) if results["left_foot"] else 0
    right_conf = results["right_foot"].get("confidence", 0) if results["right_foot"] else 0

    # Determine combined prediction
    if left_diabetic is not None and right_diabetic is not None:
        # Both feet analyzed
        if left_diabetic and right_diabetic:
            results["combined_prediction"] = "Diabetic"
            results["combined_confidence"] = round((left_conf + right_conf) / 2, 2)
            results["diagnosis_factors"].append("Both feet show diabetic indicators")
        elif left_diabetic or right_diabetic:
            # One foot shows signs - this is concerning
            results["combined_prediction"] = "Diabetic"
            results["combined_confidence"] = round(max(left_conf, right_conf) * 0.9, 2)
            affected = "left" if left_diabetic else "right"
            results["diagnosis_factors"].append(f"The {affected} foot shows diabetic indicators")
        else:
            results["combined_prediction"] = "Control"
            results["combined_confidence"] = round((left_conf + right_conf) / 2, 2)
            results["diagnosis_factors"].append("Neither foot shows diabetic indicators")

        # Factor in asymmetry
        if results["asymmetry"] and results["asymmetry"]["asymmetry_significant"]:
            results["diagnosis_factors"].append(
                f"Significant temperature asymmetry detected ({results['asymmetry']['mean_temp_difference']}°C difference)"
            )
            # Asymmetry can indicate early DPN even if individual predictions are negative
            if results["combined_prediction"] == "Control":
                results["diagnosis_factors"].append("Asymmetry warrants further clinical evaluation")

    elif left_diabetic is not None:
        results["combined_prediction"] = results["left_foot"]["prediction"]
        results["combined_confidence"] = left_conf
        results["diagnosis_factors"].append("Only left foot analyzed")
    elif right_diabetic is not None:
        results["combined_prediction"] = results["right_foot"]["prediction"]
        results["combined_confidence"] = right_conf
        results["diagnosis_factors"].append("Only right foot analyzed")

    results["is_diabetic"] = results["combined_prediction"] == "Diabetic"

    return results


# Singleton instance for API use
_classifier_instance: Optional[DPNClassifier] = None


def get_classifier(
    model_path: str = None,
    model_type: str = "cnn"
) -> DPNClassifier:
    """
    Get or create a classifier instance (singleton pattern).

    Args:
        model_path: Path to model file
        model_type: Type of model

    Returns:
        DPNClassifier instance
    """
    global _classifier_instance

    if _classifier_instance is None:
        if model_path is None:
            # Default paths
            base_path = Path(__file__).parent.parent / "checkpoints"
            if model_type == "cnn":
                model_path = base_path / "best_model.pth"
            else:
                model_path = base_path / "best_sklearn_model.joblib"

        _classifier_instance = DPNClassifier(model_path, model_type)

    return _classifier_instance


if __name__ == "__main__":
    # Test inference
    print("Testing DPN Classifier...")

    # Test with CNN
    try:
        classifier = DPNClassifier(
            model_path="../checkpoints/best_model.pth",
            model_type="cnn"
        )

        # Test with a sample image
        test_image = "../data/Control Group/CG001_M/CG001_M_L.png"
        if Path(test_image).exists():
            result = classifier.predict(test_image)
            print(f"\nPrediction result:")
            print(f"  Class: {result['prediction']}")
            print(f"  Confidence: {result['confidence']}%")
            print(f"  Probabilities: {result['probabilities']}")
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
