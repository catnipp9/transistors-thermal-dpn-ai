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

from models.model import get_model, YOLOv11DPNClassifier
from models.data_loader import get_transforms
from models.preprocessing import extract_thermal_features, extract_foot_roi


class DPNClassifier:
    """
    Unified classifier for DPN detection.
    Supports YOLOv11 (recommended), CNN (PyTorch), and classical ML (sklearn) models.

    model_type values:
        "yolo"    – YOLOv11 classification model (.pt file)  ← recommended
        "cnn"     – Custom PyTorch CNN (.pth checkpoint)
        "sklearn" – scikit-learn pipeline (.joblib file)
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "yolo",
        device: str = None
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to saved model file.
                        .pt  → YOLOv11 weights
                        .pth → CNN checkpoint
                        .joblib → sklearn pipeline
            model_type: Type of model - "yolo", "cnn", or "sklearn"
            device: Device for inference ("cuda" or "cpu"). Auto-detected if None.
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

        if self.model_type == "yolo":
            # Load YOLOv11 classification model
            self.model = YOLOv11DPNClassifier.load(str(self.model_path), device=self.device)
            print(f"Loaded YOLOv11 model from {self.model_path}")
            print(f"Running on: {self.device}")

        elif self.model_type == "cnn":
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
            raise ValueError(f"Unknown model type: {self.model_type}. Use 'yolo', 'cnn', or 'sklearn'.")

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

        Uses extract_thermal_features() — the same feature engineering applied
        during training — rather than raw flattening. This gives 54 angiosome-
        aligned features (MPA/LPA/MCA/LCA + global stats + gradients + hot/cold
        spots) instead of 10,920 raw values, matching what the saved model was
        trained on.

        Args:
            csv_path: Path to CSV file
            target_shape: Target shape for resizing

        Returns:
            Feature vector shaped (1, 54)
        """
        import pandas as pd
        from scipy.ndimage import zoom

        data = pd.read_csv(csv_path, header=None).values.astype(np.float32)

        if data.shape != target_shape:
            zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
            data = zoom(data, zoom_factors, order=1)

        features = extract_thermal_features(data)
        return features.reshape(1, -1)

    def predict(
        self,
        input_data: Union[str, Path, Image.Image, np.ndarray],
        return_proba: bool = True
    ) -> Dict:
        """
        Make a prediction on input data.

        Args:
            input_data: Image path, PIL Image, numpy array (for yolo/cnn),
                        or CSV path (for sklearn)
            return_proba: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        if self.model_type == "yolo":
            return self._predict_yolo(input_data, return_proba)
        elif self.model_type == "cnn":
            return self._predict_cnn(input_data, return_proba)
        else:
            return self._predict_sklearn(input_data, return_proba)

    def _predict_yolo(self, image: Union[str, Path, Image.Image, np.ndarray], return_proba: bool) -> Dict:
        """Make prediction using YOLOv11 classification model.

        Applies ROI crop before inference so the model always receives a
        tightly-framed foot image, regardless of source camera resolution
        (e.g. 160×120 vs the training data's 168×65 aspect ratio).
        """
        # Load to PIL if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8")).convert("RGB")

        # Crop to foot ROI — neutralises aspect-ratio mismatch from different cameras
        image = extract_foot_roi(image)

        result = self.model.predict(image)

        if not return_proba:
            result.pop("probabilities", None)

        return result

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

    # Clinical threshold: mean inter-foot temperature difference >2.2°C AND
    # mean pixel-level asymmetry >1.0°C. Requiring both prevents a single
    # hotspot from triggering a false positive asymmetry flag.
    ASYMMETRY_THRESHOLD = 2.2
    is_significant = mean_temp_diff > ASYMMETRY_THRESHOLD and mean_asymmetry > 1.0

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


def fuse_foot_predictions(
    yolo_result: Dict,
    sklearn_result: Dict,
    yolo_weight: float = 0.6
) -> Dict:
    """
    Fuse per-foot predictions from YOLOv11 (image) and sklearn (CSV temperature).

    Weighted average: YOLO gets 60% weight (97.5% accuracy on this dataset)
    and sklearn gets 40% weight as the secondary temperature-signal model.
    Both probabilities are in percent [0–100].
    """
    sklearn_weight = 1.0 - yolo_weight

    yolo_diabetic = yolo_result.get("probabilities", {}).get("Diabetic", 0.0)
    yolo_control  = yolo_result.get("probabilities", {}).get("Control",  0.0)

    sklearn_diabetic = sklearn_result.get("probabilities", {}).get("Diabetic", 0.0)
    sklearn_control  = sklearn_result.get("probabilities", {}).get("Control",  0.0)

    fused_diabetic = yolo_weight * yolo_diabetic + sklearn_weight * sklearn_diabetic
    fused_control  = yolo_weight * yolo_control  + sklearn_weight * sklearn_control

    # Normalise (each source already sums to 100, so this is a safety guard)
    total = fused_diabetic + fused_control
    if total > 0:
        fused_diabetic = fused_diabetic / total * 100
        fused_control  = fused_control  / total * 100

    is_diabetic = fused_diabetic >= fused_control
    prediction  = "Diabetic" if is_diabetic else "Control"

    return {
        "prediction":           prediction,
        "class_index":          1 if is_diabetic else 0,
        "confidence":           round(max(fused_diabetic, fused_control), 2),
        "is_diabetic":          is_diabetic,
        "probabilities": {
            "Control":  round(fused_control,  2),
            "Diabetic": round(fused_diabetic, 2),
        },
        "yolo_probabilities":    yolo_result.get("probabilities", {}),
        "sklearn_probabilities": sklearn_result.get("probabilities", {}),
        "fusion_method": (
            f"weighted_average(yolo={yolo_weight:.0%}, "
            f"sklearn={sklearn_weight:.0%})"
        ),
        "input_type": "image+csv",
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

    import tempfile

    def _predict_foot(image, csv_path, temps):
        """Run all available classifiers for one foot and fuse when both present."""
        yolo_res    = None
        sklearn_res = None

        # --- Image branch (YOLOv11) ---
        if image is not None and cnn_classifier is not None:
            yolo_res = cnn_classifier.predict(image, return_proba=True)
            yolo_res["input_type"] = "image"

        # --- Temperature branch (sklearn) ---
        if csv_path is not None and sklearn_classifier is not None:
            sklearn_res = sklearn_classifier.predict(csv_path, return_proba=True)
            sklearn_res["input_type"] = "csv"
        elif temps is not None and sklearn_classifier is not None:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                pd.DataFrame(temps).to_csv(f.name, header=False, index=False)
                sklearn_res = sklearn_classifier.predict(f.name, return_proba=True)
                sklearn_res["input_type"] = "temperature_array"
                Path(f.name).unlink(missing_ok=True)

        # --- Fusion ---
        if yolo_res is not None and sklearn_res is not None:
            return fuse_foot_predictions(yolo_res, sklearn_res)
        if yolo_res is not None:
            return yolo_res
        if sklearn_res is not None:
            return sklearn_res
        return None

    # Load left temps from CSV if not already provided (needed for asymmetry)
    if left_temps is None and left_csv is not None:
        left_temps = pd.read_csv(left_csv, header=None).values.astype(np.float32)

    # Load right temps from CSV if not already provided
    if right_temps is None and right_csv is not None:
        right_temps = pd.read_csv(right_csv, header=None).values.astype(np.float32)

    results["left_foot"]  = _predict_foot(left_image,  left_csv,  left_temps)
    results["right_foot"] = _predict_foot(right_image, right_csv, right_temps)

    # Calculate asymmetry when temperature matrices are available for both feet
    if left_temps is not None and right_temps is not None:
        results["asymmetry"] = calculate_asymmetry(left_temps, right_temps)

    # Combined diagnosis logic
    # -------------------------------------------------------------------------
    # Both feet must independently cross the diabetic threshold for a Diabetic
    # result. This prevents false positives from borderline single-foot readings.
    # If only one foot is flagged, it is noted but NOT enough alone to diagnose.
    # The model's own argmax (>50%) is used as the primary decision boundary.
    def _diabetic_prob(foot_result):
        if foot_result is None:
            return 0.0
        return foot_result.get("probabilities", {}).get("Diabetic", 0.0)

    def _is_diabetic(foot_result):
        """Flag as diabetic if Diabetic probability >= 45%.
        Slightly below 50% to catch borderline cases where the model is
        nearly split — a 48% Diabetic reading is clinically concerning
        and should not be silently classified as Control."""
        if foot_result is None:
            return None
        diabetic_prob = foot_result.get("probabilities", {}).get("Diabetic", 0.0)
        return diabetic_prob >= 45.0

    left_diabetic = _is_diabetic(results["left_foot"])
    right_diabetic = _is_diabetic(results["right_foot"])
    left_prob = _diabetic_prob(results["left_foot"])
    right_prob = _diabetic_prob(results["right_foot"])

    # Determine combined prediction
    if left_diabetic is not None and right_diabetic is not None:
        # Both feet analyzed — require BOTH to be diabetic for a positive result
        if left_diabetic and right_diabetic:
            results["combined_prediction"] = "Diabetic"
            results["combined_confidence"] = round((left_prob + right_prob) / 2, 2)
            results["diagnosis_factors"].append("Both feet show diabetic indicators")
        elif left_diabetic or right_diabetic:
            # One foot crosses threshold — but also check if combined average is high
            avg_prob = (left_prob + right_prob) / 2
            affected = "left" if left_diabetic else "right"
            affected_prob = left_prob if left_diabetic else right_prob
            # If average diabetic probability across both feet >= 40%, treat as Diabetic
            # This catches cases like 48% + 33% = 40.5% avg which is clinically concerning
            if avg_prob >= 40.0:
                results["combined_prediction"] = "Diabetic"
                results["combined_confidence"] = round(avg_prob, 2)
                results["diagnosis_factors"].append(
                    f"Combined diabetic probability ({avg_prob:.1f}%) indicates diabetic signs "
                    f"— left: {left_prob:.1f}%, right: {right_prob:.1f}%"
                )
            else:
                results["combined_prediction"] = "Control"
                results["combined_confidence"] = round(100.0 - avg_prob, 2)
                results["diagnosis_factors"].append(
                    f"The {affected} foot shows some diabetic indicators "
                    f"({affected_prob:.1f}% probability) — further evaluation recommended"
                )
        else:
            results["combined_prediction"] = "Control"
            results["combined_confidence"] = round(100.0 - (left_prob + right_prob) / 2, 2)
            results["diagnosis_factors"].append("Neither foot shows diabetic indicators")

        # Asymmetry as supporting evidence — only upgrades a borderline Control
        # to Diabetic when BOTH of these are true:
        #   1. The inter-foot mean temperature difference exceeds 2.2°C AND
        #   2. At least one foot already has a Diabetic probability > 60%
        # This prevents healthy patients with naturally asymmetric feet from
        # being falsely flagged.
        if results["asymmetry"] and results["asymmetry"]["asymmetry_significant"]:
            temp_diff = results["asymmetry"]["mean_temp_difference"]
            threshold = results["asymmetry"]["threshold_used"]
            results["diagnosis_factors"].append(
                f"Significant temperature asymmetry detected "
                f"({temp_diff:.2f}\u00b0C difference, threshold {threshold}\u00b0C)"
            )
            high_prob_foot = max(left_prob, right_prob)
            if results["combined_prediction"] == "Control" and high_prob_foot > 60.0:
                results["combined_prediction"] = "Diabetic"
                results["combined_confidence"] = round(high_prob_foot, 2)
                results["diagnosis_factors"].append(
                    "Upgraded to Diabetic: significant asymmetry combined with "
                    f"elevated diabetic probability ({high_prob_foot:.1f}%) on one foot"
                )

    elif left_diabetic is not None:
        results["combined_prediction"] = results["left_foot"]["prediction"]
        results["combined_confidence"] = round(left_prob, 2)
        results["diagnosis_factors"].append("Only left foot analyzed")
    elif right_diabetic is not None:
        results["combined_prediction"] = results["right_foot"]["prediction"]
        results["combined_confidence"] = round(right_prob, 2)
        results["diagnosis_factors"].append("Only right foot analyzed")

    results["is_diabetic"] = results["combined_prediction"] == "Diabetic"

    return results


# Singleton instance for API use
_classifier_instance: Optional[DPNClassifier] = None


def get_classifier(
    model_path: str = None,
    model_type: str = "yolo"
) -> DPNClassifier:
    """
    Get or create a classifier instance (singleton pattern).

    Defaults to the YOLOv11 model for best accuracy.  Falls back to CNN or
    sklearn if the YOLO checkpoint is not present.

    Args:
        model_path: Path to model file. Auto-detected from checkpoints/ if None.
        model_type: "yolo" (default), "cnn", or "sklearn".

    Returns:
        DPNClassifier instance
    """
    global _classifier_instance

    if _classifier_instance is None:
        if model_path is None:
            base_path = Path(__file__).parent.parent / "checkpoints"
            default_paths = {
                "yolo":    base_path / "best_yolo_model.pt",
                "cnn":     base_path / "best_model.pth",
                "sklearn": base_path / "best_sklearn_model.joblib",
            }
            # Auto-select: prefer YOLO if available, else CNN, else sklearn
            if model_type == "yolo":
                yolo_path = default_paths["yolo"]
                if not yolo_path.exists():
                    # Try Ultralytics run output path as fallback
                    alt = base_path / "yolo11_dpn" / "weights" / "best.pt"
                    yolo_path = alt if alt.exists() else yolo_path
                model_path = yolo_path
            else:
                model_path = default_paths.get(model_type, default_paths["cnn"])

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
