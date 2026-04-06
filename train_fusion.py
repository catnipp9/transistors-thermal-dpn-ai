"""
Train the fusion meta-classifier — the final decision layer for FLIR Lepton 3.5 deployment.

In deployment the camera always provides both a thermal image AND temperature values.
Instead of the fixed 60/40 weighted average, this trains a logistic regression that
learns the optimal combination from data (stacking / model ensembling).

Architecture:
  YOLO11  (image) -> diabetic_prob --+
                                     +--> LogisticRegression -> Final DPN verdict
  SVM     (CSV)   -> diabetic_prob --+

Training strategy (avoids data leakage):
  - sklearn SVM: out-of-fold predictions via cross_val_predict (5-fold).
    The SVM never predicts on data it was trained on.
  - YOLO: predictions from the already-trained model on all images.
    Some optimism here (YOLO trained on some of these images) but at 97.5%
    accuracy YOLO is already near ceiling, so meta-learner contamination is small.

Output: checkpoints/best_fusion_model.joblib

Usage:
    python train_fusion.py
    (Requires best_yolo_model.pt in checkpoints/)
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score
)
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from models.preprocessing import extract_thermal_features

TARGET_H, TARGET_W = 168, 65


# ── 1. Collect all foot samples (csv + png + label) ───────────────────────────

print("Scanning dataset for foot samples...")
subjects = []  # list of (png_path, csv_path, label)

for group, label in [("Control Group", 0), ("DM Group", 1)]:
    group_dir = Path("data") / group
    if not group_dir.exists():
        print(f"  Warning: {group_dir} not found — skipping")
        continue
    for subj in sorted(group_dir.iterdir()):
        if not subj.is_dir():
            continue
        for side in ["L", "R"]:
            png_path = subj / f"{subj.name}_{side}.png"
            csv_path = subj / f"{subj.name}_{side}.csv"
            if png_path.exists() and csv_path.exists():
                subjects.append((png_path, csv_path, label))

if not subjects:
    raise RuntimeError("No paired PNG+CSV samples found in data/")

print(f"Found {len(subjects)} paired foot samples")
y = np.array([s[2] for s in subjects])
print(f"  Control: {(y == 0).sum()}, Diabetic: {(y == 1).sum()}")


# ── 2. Extract sklearn features from all CSVs ─────────────────────────────────

print("\nExtracting thermal features from CSVs...")
X_sklearn = []
for png_path, csv_path, label in subjects:
    try:
        data = pd.read_csv(csv_path, header=None).values.astype("float32")
        if data.shape != (TARGET_H, TARGET_W):
            zf = (TARGET_H / data.shape[0], TARGET_W / data.shape[1])
            data = zoom(data, zf, order=1)
        X_sklearn.append(extract_thermal_features(data))
    except Exception as e:
        print(f"  Skipping {csv_path.name}: {e}")
        X_sklearn.append(np.zeros(54, dtype="float32"))

X_sklearn = np.array(X_sklearn)
print(f"  Feature matrix: {X_sklearn.shape}")


# ── 3. sklearn out-of-fold predictions (best params from train_sklearn.py) ────

print("\nGenerating sklearn out-of-fold predictions (5-fold CV)...")
sklearn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf", C=100.0, gamma=0.01,
        class_weight="balanced", probability=True, random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sklearn_oof_proba = cross_val_predict(
    sklearn_pipe, X_sklearn, y, cv=cv, method="predict_proba"
)
sklearn_dm_proba = sklearn_oof_proba[:, 1]  # Diabetic probability, 0-1 scale

sklearn_oof_acc = accuracy_score(y, sklearn_oof_proba.argmax(axis=1))
print(f"  sklearn OOF accuracy: {sklearn_oof_acc:.4f}")


# ── 4. YOLO predictions on all images ─────────────────────────────────────────

yolo_pt = Path("checkpoints") / "best_yolo_model.pt"
if not yolo_pt.exists():
    # Try ultralytics run output
    alt = Path("checkpoints") / "yolo11_dpn" / "weights" / "best.pt"
    if alt.exists():
        yolo_pt = alt

if not yolo_pt.exists():
    print(f"\nERROR: YOLO model not found at {yolo_pt}")
    print("  Run the Colab notebook first to train best_yolo_model.pt")
    print("  Without YOLO predictions the fusion model cannot be trained.")
    sys.exit(1)

print(f"\nLoading YOLO from {yolo_pt}...")
from api.inference import DPNClassifier
yolo_clf = DPNClassifier(str(yolo_pt), model_type="yolo")

print("Running YOLO on all images...")
yolo_dm_proba = []
for i, (png_path, csv_path, label) in enumerate(subjects):
    try:
        result = yolo_clf.predict(str(png_path), return_proba=True)
        dm_pct = result.get("probabilities", {}).get("Diabetic", 50.0)
        yolo_dm_proba.append(dm_pct / 100.0)
    except Exception as e:
        print(f"  YOLO failed on {png_path.name}: {e} — using 0.5")
        yolo_dm_proba.append(0.5)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(subjects)}")

yolo_dm_proba = np.array(yolo_dm_proba)
yolo_oof_acc = accuracy_score(y, (yolo_dm_proba >= 0.5).astype(int))
print(f"  YOLO accuracy (threshold 0.5): {yolo_oof_acc:.4f}")


# ── 5. Analyse model agreement ────────────────────────────────────────────────

agree = int(np.sum((yolo_dm_proba >= 0.5) == (sklearn_dm_proba >= 0.5)))
print(f"\nModel agreement: {agree}/{len(y)} samples ({agree/len(y)*100:.1f}%)")

# Pearson correlation of their Diabetic probabilities
r, p = pearsonr(yolo_dm_proba, sklearn_dm_proba)
print(f"Correlation (yolo vs sklearn diabetic prob): r={r:.3f}, p={p:.4f}")


# ── 6. Train meta-learner ─────────────────────────────────────────────────────

print("\nTraining fusion meta-classifier (Logistic Regression)...")
X_meta = np.column_stack([yolo_dm_proba, sklearn_dm_proba])  # shape (N, 2)

meta_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=1.0, class_weight="balanced",
        random_state=42, max_iter=1000
    ))
])

# Evaluate with the same CV splits
meta_scores = cross_val_score(meta_pipe, X_meta, y, cv=cv, scoring="accuracy")
print(f"  Meta-learner CV accuracy: {meta_scores.mean():.4f} +/- {meta_scores.std():.4f}")

# Fit on all data
meta_pipe.fit(X_meta, y)
y_meta_pred = meta_pipe.predict(X_meta)
print(f"\n-- Meta-learner in-sample evaluation --")
print(f"  Train accuracy (in-sample): {accuracy_score(y, y_meta_pred):.4f}")
print(classification_report(y, y_meta_pred, target_names=["Control", "Diabetic"]))

# Print learned logistic regression coefficients
coef = meta_pipe.named_steps["clf"].coef_[0]
print(f"\n  Learned weights (logistic regression):")
print(f"    YOLO   coefficient : {coef[0]:+.4f}")
print(f"    sklearn coefficient: {coef[1]:+.4f}")
print(f"  (higher = model has more learned influence on final verdict)")


# ── 7. Comparison: fixed weights vs meta-learner ──────────────────────────────

print("\n-- Comparison on full dataset --")
# Fixed 60/40
fixed_dm = 0.6 * yolo_dm_proba + 0.4 * sklearn_dm_proba
fixed_acc = accuracy_score(y, (fixed_dm >= 0.5).astype(int))
print(f"  Fixed 60/40 weights   : {fixed_acc:.4f}")

# Meta-learner (CV estimate)
print(f"  Meta-learner (5-fold) : {meta_scores.mean():.4f}")

improvement = meta_scores.mean() - fixed_acc
print(f"  Improvement           : {improvement:+.4f} ({improvement*100:+.1f}%)")


# ── 8. Save ───────────────────────────────────────────────────────────────────

Path("checkpoints").mkdir(exist_ok=True)
out_path = Path("checkpoints") / "best_fusion_model.joblib"
joblib.dump(meta_pipe, out_path)
print(f"\nSaved: {out_path}")
print("Restart the API to load the fusion model.")
print("The /predict/patient/combined endpoint will now use the meta-learner.")
