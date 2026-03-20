"""
Data Loader for Plantar Thermogram Dataset
Handles loading and organizing thermogram data for DPN classification
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ThermogramDataset(Dataset):
    """PyTorch Dataset for plantar thermogram images."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        use_csv: bool = False,
        foot: str = "both"  # "left", "right", or "both"
    ):
        """
        Args:
            data_dir: Path to data directory containing Control Group and DM Group
            transform: Optional torchvision transforms
            use_csv: If True, load temperature matrices from CSV; if False, load PNG images
            foot: Which foot to use - "left", "right", or "both"
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_csv = use_csv
        self.foot = foot

        self.samples = []  # List of (file_path, label, subject_id)
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        """Load all samples from Control Group and DM Group directories."""
        # Control Group (label = 0)
        control_dir = self.data_dir / "Control Group"
        if control_dir.exists():
            self._add_samples_from_group(control_dir, label=0)

        # DM Group (label = 1) - Diabetic patients
        dm_dir = self.data_dir / "DM Group"
        if dm_dir.exists():
            self._add_samples_from_group(dm_dir, label=1)

        print(f"Loaded {len(self.samples)} samples")
        print(f"  Control: {sum(1 for _, l, _ in self.samples if l == 0)}")
        print(f"  Diabetic: {sum(1 for _, l, _ in self.samples if l == 1)}")

    def _add_samples_from_group(self, group_dir: Path, label: int):
        """Add samples from a group directory."""
        for subject_folder in sorted(group_dir.iterdir()):
            if not subject_folder.is_dir():
                continue

            subject_id = subject_folder.name

            # Determine which files to load based on foot parameter
            feet_to_load = []
            if self.foot in ["left", "both"]:
                feet_to_load.append("L")
            if self.foot in ["right", "both"]:
                feet_to_load.append("R")

            for foot_side in feet_to_load:
                if self.use_csv:
                    file_path = subject_folder / f"{subject_id}_{foot_side}.csv"
                else:
                    file_path = subject_folder / f"{subject_id}_{foot_side}.png"

                if file_path.exists():
                    self.samples.append((file_path, label, subject_id))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from scipy.ndimage import zoom

        file_path, label, subject_id = self.samples[idx]
        target_h, target_w = 168, 65  # Standard thermogram size

        if self.use_csv:
            # Load temperature matrix from CSV
            data = pd.read_csv(file_path, header=None).values
            data = data.astype(np.float32)

            # Resize if shape doesn't match target
            if data.shape != (target_h, target_w):
                zoom_factors = (target_h / data.shape[0], target_w / data.shape[1])
                data = zoom(data, zoom_factors, order=1)

            # Normalize temperature values
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)

            # Add channel dimension: (H, W) -> (1, H, W)
            data = np.expand_dims(data, axis=0)
            tensor = torch.from_numpy(data)
        else:
            # Load PNG image
            image = Image.open(file_path).convert("RGB")

            if self.transform:
                tensor = self.transform(image)
            else:
                # Default transform: convert to tensor and normalize
                tensor = transforms.ToTensor()(image)

        return tensor, label

    def get_subject_ids(self) -> List[str]:
        """Return list of all subject IDs."""
        return [s[2] for s in self.samples]


def get_transforms(train: bool = True, image_size: Tuple[int, int] = (168, 65)):
    """
    Get image transforms for training or validation.

    Args:
        train: If True, include data augmentation
        image_size: Target size (height, width)
    """
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    use_csv: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders with stratified splitting.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for data loaders
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        use_csv: Whether to use CSV files instead of PNG images
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load full dataset to get indices
    full_dataset = ThermogramDataset(
        data_dir=data_dir,
        transform=None,
        use_csv=use_csv
    )

    # Get labels for stratified split
    labels = full_dataset.labels
    indices = list(range(len(full_dataset)))

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Second split: train vs val
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_val_labels
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    # Create subset datasets with appropriate transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Create new datasets with transforms
    train_dataset = ThermogramDataset(data_dir, transform=train_transform, use_csv=use_csv)
    val_dataset = ThermogramDataset(data_dir, transform=val_transform, use_csv=use_csv)
    test_dataset = ThermogramDataset(data_dir, transform=val_transform, use_csv=use_csv)

    # Create subset samplers
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def load_data_for_sklearn(
    data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    use_csv: bool = True,
    target_shape: Tuple[int, int] = (168, 65)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data in format suitable for scikit-learn models.

    Args:
        data_dir: Path to data directory
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        use_csv: Whether to use CSV (temperature) or PNG (image) data
        target_shape: Target shape (height, width) to resize all samples to

    Returns:
        X_train, X_test, y_train, y_test as numpy arrays
    """
    from scipy.ndimage import zoom

    data_dir = Path(data_dir)
    X = []
    y = []
    skipped = 0

    groups = [
        ("Control Group", 0),
        ("DM Group", 1)
    ]

    target_h, target_w = target_shape
    target_size = target_h * target_w

    for group_name, label in groups:
        group_dir = data_dir / group_name
        if not group_dir.exists():
            continue

        for subject_folder in sorted(group_dir.iterdir()):
            if not subject_folder.is_dir():
                continue

            subject_id = subject_folder.name

            for foot_side in ["L", "R"]:
                if use_csv:
                    file_path = subject_folder / f"{subject_id}_{foot_side}.csv"
                    if file_path.exists():
                        try:
                            data = pd.read_csv(file_path, header=None).values.astype(np.float32)

                            # Resize if shape doesn't match target
                            if data.shape != target_shape:
                                zoom_factors = (target_h / data.shape[0], target_w / data.shape[1])
                                data = zoom(data, zoom_factors, order=1)

                            X.append(data.flatten())
                            y.append(label)
                        except Exception as e:
                            print(f"Warning: Skipping {file_path.name} - {e}")
                            skipped += 1
                else:
                    file_path = subject_folder / f"{subject_id}_{foot_side}.png"
                    if file_path.exists():
                        try:
                            image = Image.open(file_path).convert("RGB")
                            # Resize image to consistent size
                            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
                            X.append(np.array(image).flatten())
                            y.append(label)
                        except Exception as e:
                            print(f"Warning: Skipping {file_path.name} - {e}")
                            skipped += 1

    if len(X) == 0:
        raise ValueError("No valid samples found in the dataset!")

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    print(f"  Control: {sum(y == 0)}, Diabetic: {sum(y == 1)}")
    if skipped > 0:
        print(f"  Skipped: {skipped} files due to errors")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def prepare_yolo_dataset(
    data_dir: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> str:
    """
    Build a YOLO classification dataset from the existing thermogram PNG files
    and write a dataset YAML file that Ultralytics expects.

    Directory layout created::

        <output_dir>/
          train/
            Control/   ← PNG images for class 0
            Diabetic/  ← PNG images for class 1
          val/
            Control/
            Diabetic/
          test/
            Control/
            Diabetic/
          dataset.yaml

    Args:
        data_dir: Root data directory containing "Control Group" and "DM Group".
        output_dir: Where to write the YOLO-formatted dataset.
        test_size: Fraction of images to use for test split.
        val_size: Fraction of images to use for validation split.
        random_state: Random seed for reproducible splits.

    Returns:
        Absolute path to the generated ``dataset.yaml`` file.
    """
    import shutil
    import yaml

    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Collect all (image_path, class_name) pairs
    samples: List[Tuple[Path, str]] = []

    group_map = {
        "Control Group": "Control",
        "DM Group": "Diabetic",
    }

    for group_name, class_name in group_map.items():
        group_dir = data_path / group_name
        if not group_dir.exists():
            continue
        for subject_folder in sorted(group_dir.iterdir()):
            if not subject_folder.is_dir():
                continue
            for png_file in subject_folder.glob("*.png"):
                samples.append((png_file, class_name))

    if not samples:
        raise ValueError(f"No PNG images found under {data_dir}")

    # Stratified train / val / test split
    paths = [s[0] for s in samples]
    labels_str = [s[1] for s in samples]

    # First cut off test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels_str,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_str,
    )

    # Then cut off val set from the remaining
    val_fraction = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val_labels,
    )

    splits = {
        "train": list(zip(train_paths, train_labels)),
        "val":   list(zip(val_paths,   val_labels)),
        "test":  list(zip(test_paths,  test_labels)),
    }

    print("\nYOLO dataset split:")
    for split_name, split_samples in splits.items():
        ctrl = sum(1 for _, c in split_samples if c == "Control")
        diab = sum(1 for _, c in split_samples if c == "Diabetic")
        print(f"  {split_name:5s}: {len(split_samples)} images  (Control={ctrl}, Diabetic={diab})")

    # Copy images into YOLO class sub-folders
    for split_name, split_samples in splits.items():
        for src_path, class_name in split_samples:
            dest_dir = out_path / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_dir / src_path.name)

    # Oversample minority class in train split only to address class imbalance.
    # Control (45 subjects) is heavily underrepresented vs Diabetic (122 subjects).
    # Val and test splits are left untouched — never augment evaluation data.
    train_dir = out_path / "train"
    class_counts = {
        cls: len(list((train_dir / cls).glob("*.png")))
        for cls in ["Control", "Diabetic"]
    }
    majority_count = max(class_counts.values())
    print(f"\nClass counts before oversampling: {class_counts}")
    for cls, count in class_counts.items():
        if count < majority_count:
            existing = sorted((train_dir / cls).glob("*.png"))
            copies_needed = majority_count - count
            cycle = (existing * ((copies_needed // len(existing)) + 1))[:copies_needed]
            for i, src in enumerate(cycle):
                shutil.copy2(src, train_dir / cls / f"{src.stem}_os{i}{src.suffix}")
            print(f"  Oversampled '{cls}': {count} -> {majority_count} images (+{copies_needed} copies)")

    # Write dataset YAML
    yaml_content = {
        "path": str(out_path.resolve()),
        "train": "train",
        "val":   "val",
        "test":  "test",
        "nc": 2,
        "names": ["Control", "Diabetic"],
    }
    yaml_path = out_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"\nYOLO dataset written to: {out_path}")
    print(f"Dataset YAML: {yaml_path}")
    return str(yaml_path.resolve())


if __name__ == "__main__":
    # Test the data loader
    data_dir = "../data"

    print("Testing PyTorch DataLoader...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=8
    )

    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels}")

    print("\nTesting sklearn data loader...")
    X_train, X_test, y_train, y_test = load_data_for_sklearn(data_dir)
    print(f"X_train shape: {X_train.shape}")
