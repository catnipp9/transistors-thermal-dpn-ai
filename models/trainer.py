"""
Training utilities for DPN classification models
Supports both PyTorch deep learning and scikit-learn classical ML
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class CNNTrainer:
    """Trainer class for PyTorch CNN models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_dir: str = "checkpoints"
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_acc = 0.0

        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("-" * 50)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Update learning rate
            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(save_path / "best_model.pth")
                print(f"Saved best model with val_acc: {val_acc:.4f}")

            # Early stopping
            if early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save final model
        self.save_checkpoint(save_path / "final_model.pth")

        # Save training history
        with open(save_path / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='binary'),
            'recall': recall_score(all_labels, all_preds, average='binary'),
            'f1': f1_score(all_labels, all_preds, average='binary'),
            'roc_auc': roc_auc_score(all_labels, all_probs),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'classification_report': classification_report(all_labels, all_preds,
                                                          target_names=['Control', 'Diabetic'])
        }

        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


class SklearnTrainer:
    """Trainer class for scikit-learn models."""

    def __init__(self, model):
        self.model = model
        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model."""
        print(f"Training {type(self.model).__name__}...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        self.is_fitted = True
        print(f"Training completed in {training_time:.2f} seconds")

        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")

        return {'training_time': training_time, 'train_accuracy': train_acc}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test set."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.model.predict(X_test)

        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred,
                                                          target_names=['Control', 'Diabetic'])
        }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")


def cross_validate_sklearn(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict:
    """
    Perform k-fold cross-validation for sklearn models.

    Args:
        model: sklearn model or pipeline
        X: Features
        y: Labels
        cv: Number of folds

    Returns:
        Dictionary with mean and std of metrics
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Get cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=skf)

    # Get probabilities if available
    try:
        y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
        roc_auc = roc_auc_score(y, y_proba)
    except:
        roc_auc = None

    metrics = {
        'cv_accuracy': accuracy_score(y, y_pred),
        'cv_precision': precision_score(y, y_pred, average='binary'),
        'cv_recall': recall_score(y, y_pred, average='binary'),
        'cv_f1': f1_score(y, y_pred, average='binary'),
        'cv_roc_auc': roc_auc,
        'cv_confusion_matrix': confusion_matrix(y, y_pred).tolist()
    }

    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    for key, value in metrics.items():
        if key == 'classification_report':
            print(f"\n{value}")
        elif key == 'confusion_matrix':
            print(f"\nConfusion Matrix:")
            cm = np.array(value)
            print(f"  {'':>10} Pred Ctrl  Pred DM")
            print(f"  {'True Ctrl':>10} {cm[0,0]:>8}  {cm[0,1]:>7}")
            print(f"  {'True DM':>10} {cm[1,0]:>8}  {cm[1,1]:>7}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("=" * 50)


if __name__ == "__main__":
    # Test trainers
    print("Testing trainers...")

    # Test CNN trainer
    from model import ThermogramCNN

    model = ThermogramCNN(num_classes=2)
    trainer = CNNTrainer(model, learning_rate=0.001)
    print(f"CNN Trainer initialized on device: {trainer.device}")

    # Test sklearn trainer
    from model import create_random_forest

    rf_model = create_random_forest()
    rf_trainer = SklearnTrainer(rf_model)
    print("Sklearn Trainer initialized")
