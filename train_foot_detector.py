"""
Train a One-Class SVM foot detector.

Learns what a plantar thermal foot scan looks like using only the existing
266 foot images — no negative samples needed. At inference it scores any
incoming image and rejects inputs that fall outside the learned foot
feature distribution (e.g. hands, legs, empty frames, random objects).

How it works
------------
- Extracts 54 angiosome-aligned thermal features from every foot CSV in the dataset
- Trains a One-Class SVM (RBF kernel) on those features
- The SVM learns a tight boundary around foot feature space
- At inference: score > 0 = foot (inside boundary), score < 0 = not a foot

Output: checkpoints/best_foot_detector.joblib

Usage:
    python train_foot_detector.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from models.preprocessing import extract_thermal_features

TARGET_H, TARGET_W = 168, 65


# ── 1. Load all foot CSV samples ──────────────────────────────────────────────

print("Loading all foot thermal CSV samples...")
X = []
loaded = 0
failed = 0

for group in ["Control Group", "DM Group"]:
    group_dir = Path("data") / group
    if not group_dir.exists():
        print(f"  Warning: {group_dir} not found — skipping")
        continue
    for subj in sorted(group_dir.iterdir()):
        if not subj.is_dir():
            continue
        for side in ["L", "R"]:
            csv_path = subj / f"{subj.name}_{side}.csv"
            if not csv_path.exists():
                continue
            try:
                data = pd.read_csv(csv_path, header=None).values.astype("float32")
                if data.shape != (TARGET_H, TARGET_W):
                    zf = (TARGET_H / data.shape[0], TARGET_W / data.shape[1])
                    data = zoom(data, zf, order=1)
                X.append(extract_thermal_features(data))
                loaded += 1
            except Exception as e:
                print(f"  Skipping {csv_path.name}: {e}")
                failed += 1

if not X:
    raise RuntimeError("No CSV samples found. Check your data/ folder.")

X = np.array(X)
print(f"Loaded {loaded} foot samples ({failed} failed)")
print(f"Feature matrix: {X.shape}")


# ── 2. Train One-Class SVM ────────────────────────────────────────────────────
# nu: upper bound on fraction of training points treated as outliers.
# 0.05 means at most 5% of real feet can be considered "unusual" —
# this keeps the boundary tight without being too strict.

print("\nTraining One-Class SVM foot detector...")
detector = Pipeline([
    ("scaler", StandardScaler()),
    ("ocsvm", OneClassSVM(
        kernel="rbf",
        nu=0.10,   # allow up to 10% of training feet to be "unusual" — more permissive
        gamma="scale",
    ))
])

detector.fit(X)


# ── 3. Evaluate on training data ──────────────────────────────────────────────

scores    = detector.decision_function(X)   # positive = inside foot boundary
preds     = detector.predict(X)             # +1 = foot, -1 = not foot
n_inliers = int((preds == 1).sum())
n_outliers = int((preds == -1).sum())

print(f"\nTraining set results:")
print(f"  Recognised as foot : {n_inliers}/{len(X)} ({n_inliers/len(X)*100:.1f}%)")
print(f"  Flagged as outlier : {n_outliers}/{len(X)} ({n_outliers/len(X)*100:.1f}%)")
print(f"  Decision score — min: {scores.min():.4f}  max: {scores.max():.4f}  mean: {scores.mean():.4f}")

# The threshold is 0 (OneClassSVM boundary).
# Use minimum training score as threshold: any input at least as foot-like as
# the most unusual real foot in the training set passes. Clear non-feet
# (hands, random objects) score well below the minimum real-foot score.
soft_threshold = float(scores.min()) - 0.05   # small buffer below worst training foot
print(f"  Soft threshold (min training score - 0.05): {soft_threshold:.4f}")


# ── 4. Save ───────────────────────────────────────────────────────────────────

Path("checkpoints").mkdir(exist_ok=True)
out_path = Path("checkpoints") / "best_foot_detector.joblib"

# Save both the pipeline and the soft threshold together
joblib.dump({"model": detector, "soft_threshold": soft_threshold}, out_path)
print(f"\nSaved: {out_path}")
print("Restart the API to load the foot detector.")
