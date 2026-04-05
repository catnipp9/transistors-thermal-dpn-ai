# CLAUDE.md — DPN Classification System

Guidelines for working with this codebase.

---

## Project Overview

AI system for detecting **Diabetic Peripheral Neuropathy (DPN)** from plantar thermogram images of boot feet. Uses YOLOv11 (primary) and sklearn (CSV temperature data). Dataset: 45 Control + 122 Diabetic subjects, each with left and right foot PNG + CSV.

**Current best model:** `checkpoints/best_yolo_model.pt` — `yolo11l-cls`, trained on Google Colab T4 GPU, 100 epochs, batch=32. Achieved **97.5% top-1 accuracy** on validation set.

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
  main.py               FastAPI endpoints
  inference.py          DPNClassifier, predict_patient, calculate_asymmetry
notebooks/
  train_model.ipynb     End-to-end training notebook
checkpoints/            Saved model weights (git-ignored)
```

---

## Key Design Decisions

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
- `checkpoints/best_yolo_model.pt` is git-ignored (too large for GitHub).
- Trained model is stored in Google Drive under `DPN_Checkpoints/best_yolo_model.pt`.
- To retrain: use `notebooks/train_model_colab.ipynb` on Google Colab for GPU speed.

---

## Training

### Recommended: Google Colab (GPU, ~15 min)
1. Open `notebooks/train_model_colab.ipynb` in Colab
2. Runtime → Change runtime type → T4 GPU → Save
3. Runtime → Run all
4. Download `best_yolo_model.pt` → place in `checkpoints/`

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
- `checkpoints/best_yolo_model.pt` — primary model used by API
- `checkpoints/yolo11_dpn/weights/best.pt` — Ultralytics run output

---

## Mobile App Connection

The mobile app connects to the FastAPI server via HTTP. The main endpoint is:
```
POST /predict/patient/images  — accepts left_foot + right_foot PNG files
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
| `Classification datasets must be a directory` | Old bug (fixed) — YAML path passed instead of dir | Already handled in `model.py` |
| `CUDA not available` | No GPU — training runs on CPU | Normal; training is slower but works |
| False positives (Control → Diabetic) | Asymmetry threshold too loose or soft threshold | See `predict_patient` logic in `inference.py` |
| False negatives (Diabetic → Control) | Model underfit or class imbalance | Delete `yolo_dataset/`, retrain with oversampling |

---

## Improving Accuracy

If the model is misclassifying:

1. **Retrain from scratch** after deleting `checkpoints/yolo_dataset/` — oversampling only applies to a fresh dataset build.
2. **Use more epochs** — increase to 100 with `patience=15` in the notebook.
3. **Try a larger variant** — change `yolo11m-cls` → `yolo11l-cls` for more model capacity.
4. **Check class balance** — the notebook prints class counts before/after oversampling. Both should be equal in the train split.
5. **Do not lower the diabetic threshold below 50%** — the current logic uses the model's own argmax. Lowering it causes false positives on Control subjects.
