"""
Preprocessing utilities for plantar thermogram data
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import cv2


def normalize_temperature(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize temperature data.

    Args:
        data: Temperature matrix (H, W) or flattened array
        method: "minmax" for 0-1 scaling, "zscore" for standardization

    Returns:
        Normalized data with same shape
    """
    original_shape = data.shape
    flat = data.flatten().reshape(-1, 1)

    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized = scaler.fit_transform(flat)
    return normalized.reshape(original_shape)


def extract_roi(
    image: np.ndarray,
    threshold: float = 0.1
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract Region of Interest (foot region) from thermogram.

    Args:
        image: Input image array (H, W) or (H, W, C)
        threshold: Threshold for foreground detection

    Returns:
        Cropped image and bounding box (x, y, w, h)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    # Threshold to find foreground
    _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (foot)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        cropped = image[y:y+h, x:x+w]
        return cropped, (x, y, w, h)

    return image, (0, 0, image.shape[1], image.shape[0])


def extract_statistical_features(temperature_matrix: np.ndarray) -> dict:
    """
    Extract statistical features from temperature matrix.

    Args:
        temperature_matrix: 2D array of temperature values

    Returns:
        Dictionary of statistical features
    """
    flat = temperature_matrix.flatten()

    features = {
        "mean": np.mean(flat),
        "std": np.std(flat),
        "min": np.min(flat),
        "max": np.max(flat),
        "range": np.max(flat) - np.min(flat),
        "median": np.median(flat),
        "q25": np.percentile(flat, 25),
        "q75": np.percentile(flat, 75),
        "iqr": np.percentile(flat, 75) - np.percentile(flat, 25),
        "skewness": pd.Series(flat).skew(),
        "kurtosis": pd.Series(flat).kurtosis(),
        "variance": np.var(flat),
    }

    # Gradient features (temperature variations)
    grad_y, grad_x = np.gradient(temperature_matrix)
    features["gradient_mean"] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    features["gradient_max"] = np.max(np.sqrt(grad_x**2 + grad_y**2))

    return features


def extract_region_features(
    temperature_matrix: np.ndarray,
    n_regions: int = 6
) -> dict:
    """
    Divide foot into regions and extract features from each.
    This simulates angiosome-based analysis.

    Args:
        temperature_matrix: 2D temperature array
        n_regions: Number of regions to divide into

    Returns:
        Dictionary of region-based features
    """
    h, w = temperature_matrix.shape
    region_h = h // n_regions

    features = {}

    for i in range(n_regions):
        start_y = i * region_h
        end_y = start_y + region_h if i < n_regions - 1 else h

        region = temperature_matrix[start_y:end_y, :]

        features[f"region_{i}_mean"] = np.mean(region)
        features[f"region_{i}_std"] = np.std(region)
        features[f"region_{i}_range"] = np.max(region) - np.min(region)

    # Temperature differences between regions
    for i in range(n_regions - 1):
        diff = features[f"region_{i}_mean"] - features[f"region_{i+1}_mean"]
        features[f"region_diff_{i}_{i+1}"] = diff

    return features


def calculate_asymmetry_features(
    left_temp: np.ndarray,
    right_temp: np.ndarray
) -> dict:
    """
    Calculate temperature asymmetry between left and right foot.
    Asymmetry is an important indicator in DPN detection.

    Args:
        left_temp: Temperature matrix for left foot
        right_temp: Temperature matrix for right foot

    Returns:
        Dictionary of asymmetry features
    """
    # Ensure same shape
    if left_temp.shape != right_temp.shape:
        # Resize to match
        min_h = min(left_temp.shape[0], right_temp.shape[0])
        min_w = min(left_temp.shape[1], right_temp.shape[1])
        left_temp = cv2.resize(left_temp, (min_w, min_h))
        right_temp = cv2.resize(right_temp, (min_w, min_h))

    # Flip right foot for alignment
    right_flipped = np.fliplr(right_temp)

    diff = np.abs(left_temp - right_flipped)

    features = {
        "asymmetry_mean": np.mean(diff),
        "asymmetry_max": np.max(diff),
        "asymmetry_std": np.std(diff),
        "asymmetry_sum": np.sum(diff),
        "mean_temp_diff": np.mean(left_temp) - np.mean(right_temp),
    }

    return features


def extract_all_features(
    temperature_matrix: np.ndarray,
    include_raw: bool = False
) -> np.ndarray:
    """
    Extract all features from a temperature matrix.

    Args:
        temperature_matrix: 2D temperature array
        include_raw: Whether to include flattened raw values

    Returns:
        1D feature vector
    """
    features = {}

    # Statistical features
    features.update(extract_statistical_features(temperature_matrix))

    # Region features
    features.update(extract_region_features(temperature_matrix))

    # Convert to array
    feature_vector = np.array(list(features.values()))

    if include_raw:
        # Add normalized flattened matrix
        normalized = normalize_temperature(temperature_matrix)
        feature_vector = np.concatenate([feature_vector, normalized.flatten()])

    return feature_vector


def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Apply PCA for dimensionality reduction.

    Args:
        X_train: Training features
        X_test: Test features
        n_components: Number of components or variance to retain

    Returns:
        Transformed X_train, X_test, and fitted PCA object
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} features")
    print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2%}")

    return X_train_pca, X_test_pca, pca


def select_best_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 100
) -> Tuple[np.ndarray, np.ndarray, SelectKBest]:
    """
    Select top k features using ANOVA F-value.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        k: Number of features to select

    Returns:
        Transformed X_train, X_test, and fitted selector
    """
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    print(f"Feature selection: {X_train.shape[1]} -> {X_train_selected.shape[1]} features")

    return X_train_selected, X_test_selected, selector


def augment_temperature_data(
    temperature_matrix: np.ndarray,
    augmentations: List[str] = None
) -> List[np.ndarray]:
    """
    Apply data augmentation to temperature matrix.

    Args:
        temperature_matrix: Original temperature data
        augmentations: List of augmentation types to apply

    Returns:
        List of augmented matrices (including original)
    """
    if augmentations is None:
        augmentations = ["flip_h", "rotate", "noise"]

    results = [temperature_matrix]  # Include original

    for aug in augmentations:
        if aug == "flip_h":
            results.append(np.fliplr(temperature_matrix))
        elif aug == "flip_v":
            results.append(np.flipud(temperature_matrix))
        elif aug == "rotate":
            # Small rotation (-5 to 5 degrees)
            angle = np.random.uniform(-5, 5)
            h, w = temperature_matrix.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(temperature_matrix, M, (w, h))
            results.append(rotated)
        elif aug == "noise":
            # Add small Gaussian noise
            noise = np.random.normal(0, 0.01, temperature_matrix.shape)
            noisy = temperature_matrix + noise
            results.append(np.clip(noisy, 0, 1))

    return results


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing utilities...")

    # Create dummy temperature matrix
    temp_matrix = np.random.rand(168, 65) * 10 + 25  # 25-35°C range

    # Test normalization
    normalized = normalize_temperature(temp_matrix)
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Test feature extraction
    features = extract_all_features(temp_matrix)
    print(f"Extracted {len(features)} features")

    # Test augmentation
    augmented = augment_temperature_data(normalized)
    print(f"Generated {len(augmented)} samples from augmentation")
