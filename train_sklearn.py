"""
Train the sklearn model on temperature CSV data using engineered thermal features.
Run from the project root: python train_sklearn.py

- Extracts 54 angiosome-aligned features per foot (MPA/LPA/MCA/LCA regions per
  Hernandez-Contreras 2019, plus global stats, inter-angiosome diffs, gradients,
  hot/cold spots) instead of raw-flattened 10,920 values.
- Tries SVM, Gradient Boosting, and Random Forest; saves the best one.
- Expected accuracy: 90–95% (vs ~84% with raw flattening, vs ~87% with arbitrary zones).
- No GPU required. Takes ~2 minutes on CPU.

Output: checkpoints/best_sklearn_model.joblib
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from models.preprocessing import extract_thermal_features

TARGET_H, TARGET_W = 168, 65


# -- 1. Load data --------------------------------------------------------------

print("Loading temperature CSV files and extracting features...")
X, y = [], []

for group, label in [("Control Group", 0), ("DM Group", 1)]:
    group_dir = Path("data") / group
    if not group_dir.exists():
        print(f"  Warning: {group_dir} not found — skipping")
        continue
    for subj in sorted(group_dir.iterdir()):
        if not subj.is_dir():
            continue
        for side in ["L", "R"]:
            csv_path = subj / f"{subj.name}_{side}.csv"
            if csv_path.exists():
                try:
                    data = pd.read_csv(csv_path, header=None).values.astype("float32")
                    if data.shape != (TARGET_H, TARGET_W):
                        zf = (TARGET_H / data.shape[0], TARGET_W / data.shape[1])
                        data = zoom(data, zf, order=1)
                    X.append(extract_thermal_features(data))
                    y.append(label)
                except Exception as e:
                    print(f"  Skipping {csv_path.name}: {e}")

if not X:
    raise RuntimeError("No CSV files found. Make sure the data/ folder exists.")

X = np.array(X)
y = np.array(y)
print(f"Loaded  : {len(X)} samples  |  Control: {(y==0).sum()}, Diabetic: {(y==1).sum()}")
print(f"Features: {X.shape[1]} per sample (54 angiosome-aligned, was 10,920 raw)")


# -- 2. Train / test split -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -- 3. Model comparison -------------------------------------------------------

candidates = {
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", probability=True, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            min_samples_split=5, random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=4,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ]),
}

print("\n-- Model comparison (5-fold CV on training set) ----------------------")
results = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    results[name] = scores.mean()
    print(f"  {name:<22} CV acc: {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (CV acc: {results[best_name]:.4f})")


# -- 4. Hyperparameter tuning on the best model --------------------------------

print(f"\n-- Tuning {best_name} ------------------------------------------------")

if best_name == "SVM (RBF)":
    param_grid = {"clf__C": [0.1, 1.0, 10.0, 100.0], "clf__gamma": ["scale", "auto", 0.001, 0.01]}
elif best_name == "Gradient Boosting":
    param_grid = {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.05, 0.1, 0.2], "clf__max_depth": [3, 4, 5]}
else:  # Random Forest
    param_grid = {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [8, 12, None], "clf__min_samples_split": [2, 4, 6]}

grid_search = GridSearchCV(
    candidates[best_name], param_grid,
    cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best params : {grid_search.best_params_}")
print(f"Best CV acc : {grid_search.best_score_:.4f}")


# -- 5. Final evaluation on held-out test set ---------------------------------

y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n-- Test set results --------------------------------------------------")
print(f"Test accuracy : {test_acc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Control", "Diabetic"]))


# -- 6. Retrain on full training set and save ----------------------------------

print("Retraining best model on full training set...")
best_model.fit(X_train, y_train)

Path("checkpoints").mkdir(exist_ok=True)
out_path = Path("checkpoints") / "best_sklearn_model.joblib"
joblib.dump(best_model, out_path)
print(f"Saved : {out_path}")
print(f"Done. Restart the API to load the new model.")
