# CLAUDE.md — DPN Classification System

Guidelines for working with this codebase.

---

## Project Overview

AI system for detecting **Diabetic Peripheral Neuropathy (DPN)** from plantar thermogram images of boot feet. Uses YOLOv11 (primary) and sklearn (CSV temperature data). Dataset: 45 Control + 122 Diabetic subjects, each with left and right foot PNG + CSV.

**Current best models:**
- `checkpoints/best_yolo_model.pt` — `yolo11l-cls`, Colab T4 GPU, 100 epochs, batch=32. **97.5% top-1 accuracy** on thermal images.
- `checkpoints/best_sklearn_model.joblib` — SVM RBF pipeline (C=100, gamma=0.01), trained on 54 angiosome-aligned features (MPA/LPA/MCA/LCA per Hernandez-Contreras 2019). **89% test accuracy**. Required for the combined `/predict/patient/combined` endpoint.

---

## Architecture

```
data/                   Raw thermogram data (PNG + CSV per foot per subject)
models/
  model.py              YOLOv11DPNClassifier, CNN architectures, sklearn factories
  data_loader.py        ThermogramDataset, prepare_yolo_dataset (with oversampling)
  trainer.py            YOLOTrainer, CNNTrainer, SklearnTrainer
  preprocessing.py      Feature extraction, normalization, augmentation
api/
  main.py               FastAPI endpoints (including /predict/patient/combined)
  inference.py          DPNClassifier, fuse_foot_predictions, predict_patient, calculate_asymmetry
notebooks/
  train_model.ipynb       Local end-to-end training (YOLO + sklearn)
  train_model_colab.ipynb Colab GPU training (YOLO + sklearn)
checkpoints/            Saved model weights (git-ignored)
```

---

## Key Design Decisions

### Dual-modality fusion (`api/inference.py → fuse_foot_predictions`)
- When both a PNG image AND a CSV file are provided for a foot, **both models run independently** and their probabilities are fused via weighted average:
  - **YOLOv11 weight: 60%** (image classifier — 97.5% accuracy)
  - **sklearn weight: 40%** (temperature classifier — 89% accuracy)
- sklearn features use paper-aligned angiosome boundaries (Hernandez-Contreras 2019):
  - MPA (Medial Plantar Artery) = top 60% rows × inner 35% cols
  - LPA (Lateral Plantar Artery) = top 60% rows × outer 65% cols  ← top discriminator
  - MCA (Medial Calcaneal Artery) = bottom 40% rows × inner 35% cols
  - LCA (Lateral Calcaneal Artery) = bottom 40% rows × outer 65% cols  ← 2nd discriminator
  - 54 total features: 12 global + 24 angiosome + 6 inter-angiosome diffs + 3 gradient + 6 hot/cold + 3 forefoot/hindfoot
- Result dict includes `yolo_probabilities`, `sklearn_probabilities`, and `fusion_method` for traceability.
- Falls back to single-modality automatically when only one input is available.
- The `_predict_foot` inner function inside `predict_patient` handles this logic.

### Dual-foot diagnosis logic (`api/inference.py → predict_patient`)
- **Both feet must independently predict Diabetic** for a positive result.
- A single diabetic foot is flagged as "further evaluation recommended" but does NOT produce a Diabetic diagnosis. This prevents false positives.
- Asymmetry only upgrades a Control result to Diabetic when **both** conditions are true: mean temp difference >2.2°C AND the higher-probability foot exceeds 60% Diabetic probability.
- Asymmetry significance requires `mean_temp_diff > 2.2°C AND mean_asymmetry > 1.0°C` (conjunctive — prevents single hotspots from triggering false flags).

### Class imbalance fix (`models/data_loader.py → prepare_yolo_dataset`)
- Control class is oversampled in the train split only (not val/test) until both classes have equal image counts.
- Delete `checkpoints/yolo_dataset/` before rerunning to get a fresh oversampled build.

### Model variant
- Currently trained: `yolo11l-cls` (large) — 97.5% accuracy, trained on Colab GPU.
- Default code default: `yolo11m-cls` (medium). Change via `YOLOv11DPNClassifier(variant='yolo11l-cls')`.
- For faster CPU training use `yolo11s-cls`.

### Model file storage
- Both `best_yolo_model.pt` and `best_sklearn_model.joblib` are git-ignored (binary files).
- After Colab training, both are saved to Google Drive under `DPN_Checkpoints/`.
- To retrain: use `notebooks/train_model_colab.ipynb` on Google Colab (trains both models).

---

## Training

### Recommended: Google Colab (GPU, ~15–20 min)
1. Open `notebooks/train_model_colab.ipynb` in Colab
2. Runtime → Change runtime type → T4 GPU → Save
3. Runtime → Run all
4. Download **both** `best_yolo_model.pt` and `best_sklearn_model.joblib` → place in `checkpoints/`

### Local training (CPU, ~3 hrs)
```powershell
# 1. Rebuild dataset (required if you changed data_loader.py)
Remove-Item -Recurse -Force checkpoints\yolo_dataset

# 2. Run notebook
jupyter notebook notebooks/train_model.ipynb
```

Or train from Python directly:
```python
from models.data_loader import prepare_yolo_dataset
from models.model import YOLOv11DPNClassifier
from models.trainer import YOLOTrainer

yaml_path = prepare_yolo_dataset("data/", "checkpoints/yolo_dataset")
model = YOLOv11DPNClassifier(variant='yolo11l-cls')
trainer = YOLOTrainer(model, save_dir="checkpoints")
trainer.train(yaml_path, epochs=100, imgsz=224, batch=16, patience=20)
trainer.save_best_checkpoint()
```

Expected output files after training:
- `checkpoints/best_yolo_model.pt` — YOLOv11 image model (primary)
- `checkpoints/best_sklearn_model.joblib` — SVM temperature model (required for combined endpoint)
- `checkpoints/yolo11_dpn/weights/best.pt` — Ultralytics run output

---

## Mobile App Connection

The mobile app connects to the FastAPI server via HTTP. The recommended endpoint is:
```
POST /predict/patient/combined  — accepts left_foot_image + right_foot_image (PNG) + left_foot_csv + right_foot_csv
```
Falls back to images-only if CSV data is unavailable:
```
POST /predict/patient/images    — accepts left_foot + right_foot PNG files
```

For local testing (same WiFi):
```
http://<PC_LOCAL_IP>:8000/predict/patient/images
```

For production (deployed): a permanent server URL is needed (Railway/Render).
The API is **not yet deployed** — currently runs locally only.

Key response fields the app should use:
- `is_diabetic` — boolean, main result
- `combined_prediction` — "Diabetic" or "Control"
- `combined_confidence` — float, percentage
- `diagnosis_factors` — array of strings, explanation

---

## Running the API

```bash
uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

---

## Testing

```bash
python test_models.py
```

---

## Git Workflow

### Commit
```bash
git add <specific files>
git commit -m "your message"
```

Never use `git add .` or `git add -A` — the `checkpoints/` folder contains large binary model files that should not be committed (already in `.gitignore`).

### Push
```bash
git push origin main
```

### Typical workflow after making changes

```bash
# 1. Check what changed
git status
git diff

# 2. Stage only source files (not checkpoints or data)
git add models/model.py models/data_loader.py api/inference.py api/main.py

# 3. Commit
git commit -m "describe what changed and why"

# 4. Push
git push origin main
```

---

## What NOT to commit

- `checkpoints/` — model weights (.pt, .pth, .joblib) and training artifacts
- `data/` — raw thermogram dataset
- `notebooks/.ipynb_checkpoints/`
- Any `.env` files

These are all covered by `.gitignore`.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Image model not loaded` | No `best_yolo_model.pt` in checkpoints | Run training notebook first |
| `sklearn model not loaded` | No `best_sklearn_model.joblib` in checkpoints | Run sklearn training cells in Colab or local notebook |
| 503 on `/predict/patient/combined` | Either model file missing | Both `.pt` and `.joblib` must exist in `checkpoints/` |
| `Classification datasets must be a directory` | Old bug (fixed) | Already handled in `model.py` |
| `CUDA not available` | No GPU | Normal — training works on CPU, just slower |
| False positives (Control → Diabetic) | Asymmetry threshold or soft threshold | See `predict_patient` in `inference.py` |
| False negatives (Diabetic → Control) | Model underfit or class imbalance | Delete `yolo_dataset/`, retrain with oversampling |

---

## Improving Accuracy

If the model is misclassifying:

1. **Retrain from scratch** after deleting `checkpoints/yolo_dataset/` — oversampling only applies to a fresh dataset build.
2. **Use more epochs** — increase to 100 with `patience=15` in the notebook.
3. **Try a larger variant** — change `yolo11m-cls` → `yolo11l-cls` for more model capacity.
4. **Check class balance** — the notebook prints class counts before/after oversampling. Both should be equal in the train split.
5. **Do not lower the diabetic threshold below 50%** — the current logic uses the model's own argmax. Lowering it causes false positives on Control subjects.
