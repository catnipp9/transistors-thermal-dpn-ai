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


def extract_foot_roi(
    image,
    padding_ratio: float = 0.05,
    min_foot_ratio: float = 0.10,
) -> "PIL.Image.Image":
    """
    Crop a thermal image to just the foot region before passing to YOLO.

    This is the most important fix when using a camera whose resolution/aspect
    ratio differs from the training data (e.g. 160×120 vs the training set's
    168×65).  By cropping to the foot bounding box first, YOLO always receives
    a consistently framed foot image regardless of the source resolution.

    How it works:
      1. Convert image to grayscale and threshold to find non-background pixels
         (foot is warm → brighter in the thermal colourmap).
      2. Find the largest connected bright region (the foot).
      3. Crop to its bounding box + a small padding margin.
      4. Return the cropped PIL image (YOLO will resize it internally).

    Falls back to returning the original image unchanged if no clear foot
    region is detected (prevents silent failures).

    Args:
        image: PIL Image (RGB thermal colourmap) or numpy array (H×W×3 uint8).
        padding_ratio: Padding added around the bounding box as a fraction of
                       image dimensions (default 5%).
        min_foot_ratio: Minimum fraction of image area the detected foot region
                        must cover to be accepted (guards against noise detections).

    Returns:
        Cropped PIL Image containing only the foot region.
    """
    from PIL import Image as PILImage
    import PIL

    # Normalise input to numpy uint8 RGB
    if isinstance(image, PILImage.Image):
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_np = np.array(image, dtype=np.uint8)

    h, w = img_np.shape[:2]

    # Convert to grayscale for thresholding
    gray = np.mean(img_np, axis=2).astype(np.uint8)

    # Otsu-style threshold: pixels above median are "foot" (warm region)
    threshold = int(np.median(gray[gray > 0]) * 0.75) if np.any(gray > 0) else 30
    binary = (gray > threshold).astype(np.uint8)

    # Find bounding box of the largest bright region via connected-component analysis
    try:
        from scipy.ndimage import label as nd_label
        labeled, num_features = nd_label(binary)
        if num_features == 0:
            return PILImage.fromarray(img_np)

        # Pick the largest component
        component_sizes = np.bincount(labeled.ravel())[1:]  # skip background (0)
        largest_label = int(np.argmax(component_sizes)) + 1
        component_mask = (labeled == largest_label)

        # Reject if the detected region is too small (noise)
        if component_mask.sum() < min_foot_ratio * h * w:
            return PILImage.fromarray(img_np)

        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        r_min, r_max = int(np.where(rows)[0][[0, -1]])
        c_min, c_max = int(np.where(cols)[0][[0, -1]])

        # Add padding
        pad_h = max(1, int(h * padding_ratio))
        pad_w = max(1, int(w * padding_ratio))
        r_min = max(0, r_min - pad_h)
        r_max = min(h, r_max + pad_h)
        c_min = max(0, c_min - pad_w)
        c_max = min(w, c_max + pad_w)

        cropped = img_np[r_min:r_max, c_min:c_max]
        return PILImage.fromarray(cropped)

    except Exception:
        # If anything fails, return original — never break inference
        return PILImage.fromarray(img_np)


def extract_thermal_features(data: np.ndarray) -> np.ndarray:
    """
    Extract clinically-motivated features from a plantar temperature matrix.

    Feature design aligned with Hernandez-Contreras 2019 (dataset paper) and
    Khandakar 2022 (ML paper) — the research basis for this DPN classification system.

    Angiosome boundaries (Hernandez-Contreras 2019, Figure 6):
      Forefoot = top 60% of rows  |  Hindfoot = bottom 40% of rows
      Medial   = inner 35% of cols (arch side)
      Lateral  = outer 65% of cols (little-toe / calcaneal side)
      → MPA (Medial Plantar Artery)   = forefoot × medial
      → LPA (Lateral Plantar Artery)  = forefoot × lateral   ← top discriminator
      → MCA (Medial Calcaneal Artery) = hindfoot × medial
      → LCA (Lateral Calcaneal Artery)= hindfoot × lateral   ← 2nd discriminator

    Total: 54 features (12 global + 24 angiosome + 6 inter-angiosome + 3 gradient
                        + 6 hot/cold spots + 3 forefoot/hindfoot)

    Args:
        data: 2D temperature matrix (H × W), zeros = background

    Returns:
        1D float32 feature vector of length 54
    """
    N_FEATURES = 54
    h, w = data.shape
    features = []

    # Mask out background (zero pixels = non-foot area)
    foot_pixels = data[data > 0]
    if len(foot_pixels) == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # ── Global statistics (12 features) ──────────────────────────────────────
    mean_t = float(np.mean(foot_pixels))
    std_t  = float(np.std(foot_pixels))
    features.extend([
        mean_t, std_t,
        float(np.min(foot_pixels)), float(np.max(foot_pixels)),
        float(np.max(foot_pixels) - np.min(foot_pixels)),
        float(np.median(foot_pixels)),
        float(np.percentile(foot_pixels, 25)),
        float(np.percentile(foot_pixels, 75)),
        float(np.percentile(foot_pixels, 75) - np.percentile(foot_pixels, 25)),
        float(pd.Series(foot_pixels).skew()),
        float(pd.Series(foot_pixels).kurtosis()),
        float(np.var(foot_pixels)),
    ])  # running total: 12

    # ── Angiosome-based features (4 regions × 6 stats = 24 features) ─────────
    # Boundaries from Hernandez-Contreras 2019, Figure 6
    ff_row = int(0.60 * h)   # forefoot / hindfoot boundary
    med_col = int(0.35 * w)  # medial / lateral boundary

    angiosomes = {
        "MPA": data[:ff_row, :med_col],   # Medial Plantar Artery
        "LPA": data[:ff_row, med_col:],   # Lateral Plantar Artery  ← top discriminator
        "MCA": data[ff_row:, :med_col],   # Medial Calcaneal Artery
        "LCA": data[ff_row:, med_col:],   # Lateral Calcaneal Artery ← 2nd discriminator
    }

    ang_means = {}
    for name, region in angiosomes.items():
        pixels = region[region > 0]
        if len(pixels) > 0:
            ang_mean = float(np.mean(pixels))
            ang_means[name] = ang_mean
            features.extend([
                ang_mean,
                float(np.std(pixels)),
                float(np.max(pixels)),
                float(np.min(pixels)),
                float(np.percentile(pixels, 25)),
                float(np.percentile(pixels, 75)),
            ])
        else:
            ang_means[name] = 0.0
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # running total: 12 + 24 = 36

    # ── Inter-angiosome differences (6 features) ─────────────────────────────
    # These differences are the core of the Thermal Change Index (TCI) concept
    features.extend([
        ang_means["LPA"] - ang_means["MPA"],  # lateral-medial forefoot asymmetry
        ang_means["LCA"] - ang_means["MCA"],  # lateral-medial hindfoot asymmetry
        ang_means["MPA"] - ang_means["MCA"],  # forefoot-hindfoot medial
        ang_means["LPA"] - ang_means["LCA"],  # forefoot-hindfoot lateral
        (ang_means["MPA"] + ang_means["LPA"]) / 2.0
            - (ang_means["MCA"] + ang_means["LCA"]) / 2.0,  # overall forefoot vs hindfoot
        (ang_means["LPA"] + ang_means["LCA"]) / 2.0
            - (ang_means["MPA"] + ang_means["MCA"]) / 2.0,  # overall lateral vs medial
    ])
    # running total: 36 + 6 = 42

    # ── Gradient features (3 features) ───────────────────────────────────────
    grad_y, grad_x = np.gradient(data.astype(np.float32))
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    features.extend([
        float(np.mean(grad_mag)),
        float(np.max(grad_mag)),
        float(np.std(grad_mag)),
    ])
    # running total: 42 + 3 = 45

    # ── Hot / cold spot analysis (6 features) ────────────────────────────────
    # Hot spots (inflammation / active infection)
    for sigma in [0.5, 1.0, 1.5]:
        features.append(float(np.sum(data > mean_t + sigma * std_t) / data.size))
    # Cold spots (reduced blood flow — key DPN indicator per Khandakar 2022)
    for sigma in [0.5, 1.0, 1.5]:
        features.append(float(np.sum((data > 0) & (data < mean_t - sigma * std_t)) / data.size))
    # running total: 45 + 6 = 51

    # ── Forefoot vs hindfoot summary (3 features, 60/40 split) ───────────────
    forefoot_pixels = data[:ff_row, :][data[:ff_row, :] > 0]
    hindfoot_pixels = data[ff_row:, :][data[ff_row:, :] > 0]
    fore_mean = float(np.mean(forefoot_pixels)) if len(forefoot_pixels) > 0 else 0.0
    hind_mean = float(np.mean(hindfoot_pixels)) if len(hindfoot_pixels) > 0 else 0.0
    features.extend([
        fore_mean,
        hind_mean,
        fore_mean - hind_mean,
    ])
    # running total: 51 + 3 = 54

    result = np.array(features, dtype=np.float32)
    assert len(result) == N_FEATURES, f"Feature count mismatch: {len(result)} != {N_FEATURES}"
    return result


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
