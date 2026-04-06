# DPN Classification System

AI-powered Diabetic Peripheral Neuropathy (DPN) detection from plantar thermograms.

---

## What the AI Does

This system classifies whether a patient has **Diabetic Peripheral Neuropathy (DPN)** by analyzing thermal images and/or temperature readings of the soles of their feet (plantar thermograms).

It takes input from **both feet** and provides:
- Individual classification per foot (Control vs Diabetic)
- Temperature asymmetry analysis between left and right feet
- A combined diagnosis with confidence scores

A temperature difference greater than **2.2 degrees Celsius** between feet is flagged as clinically significant, as asymmetry is a known indicator of nerve damage in DPN.

---

## Model Performance

| Model | Input | Accuracy | Role |
|-------|-------|----------|------|
| **YOLOv11 large** | Thermal image (PNG) | **97.5%** | Primary (60% fusion weight) |
| SVM RBF | Temperature CSV | **89%** | Secondary (40% fusion weight) |
| CNN (legacy) | Thermal image (PNG) | 88.68% | Fallback if YOLO unavailable |

> The combined `/predict/patient/combined` endpoint fuses both models — YOLOv11 reads the image, SVM reads the temperature CSV, and their weighted probabilities determine the final diagnosis.
>
> Current best image model: `checkpoints/best_yolo_model.pt` — YOLOv11 large, 100 epochs, Colab T4 GPU.

---

## AI Models Used

### 1. YOLOv11 Classification (Primary — Recommended)
- **Framework**: [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv11
- **Variant used**: `yolo11l-cls` (large)
- **Accuracy**: **97.5% top-1** on validation set
- **Trained on**: Google Colab T4 GPU, 100 epochs, batch=32, imgsz=224
- **Why**: Higher accuracy than the custom CNN due to pre-trained ImageNet weights, advanced augmentation pipeline, and cosine LR scheduling. Oversampling applied to fix class imbalance (45 Control vs 122 Diabetic).
- **Input**: RGB thermal images (PNG/JPG) — auto-resized to 224×224
- **Output**: Binary classification (Control vs Diabetic) with confidence percentage
- **Saved checkpoint**: `checkpoints/best_yolo_model.pt` (git-ignored, stored separately)

**Available size variants** (tradeoff: speed vs accuracy):

| Variant | Key | Notes |
|---------|-----|-------|
| `yolo11n-cls` | `yolo11n` | Nano — fastest |
| `yolo11s-cls` | `yolo11s` | Small — fastest on CPU |
| `yolo11m-cls` | `yolo11m` | Medium — default |
| `yolo11l-cls` | `yolo11l` | Large — **currently trained, best accuracy** |
| `yolo11x-cls` | `yolo11x` | Extra-large — most accurate |

### 2. CNN (Legacy — Thermal Images)
- **Framework**: PyTorch
- **Architecture**: LightweightCNN (custom-built)
  - 3 convolutional blocks with batch normalization and ReLU activation
  - Adaptive average pooling for flexible input sizes
  - Fully connected classifier with dropout regularization
- **Input**: RGB thermal images (PNG/JPG)
- **Output**: Binary classification (Control vs Diabetic) with confidence percentage
- **Saved checkpoint**: `checkpoints/best_model.pth`

### 3. Classical Machine Learning Models (For Temperature CSV Values)
- **Framework**: scikit-learn
- **Models trained and compared**:
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
  - Logistic Regression
- **Input**: CSV temperature matrices (168×65) — 54 angiosome-aligned features extracted (MPA/LPA/MCA/LCA regions per Hernandez-Contreras 2019)
- **Output**: Binary classification (Control vs Diabetic) with confidence percentage
- **Saved checkpoint**: `checkpoints/best_sklearn_model.joblib`

### 4. Dual-Modality Fusion (New)
- When both a thermal image **and** a temperature CSV are provided for a foot, both models run independently and their probabilities are fused:
  - **YOLOv11 weight: 60%** — image-based, highest accuracy (97.5%)
  - **sklearn weight: 40%** — temperature-based, secondary signal (89%)
- The fused probability is used for all downstream diagnosis logic
- Falls back gracefully to single-modality when only one input type is available
- Per-foot response includes `yolo_probabilities`, `sklearn_probabilities`, and `fusion_method` for transparency

### 5. Dual-Foot Diagnosis Logic
- **Both feet must independently predict Diabetic** for a Diabetic result
- A single diabetic foot is flagged as "further evaluation recommended" — not a positive diagnosis
- Asymmetry only upgrades a Control result when: mean temp diff >2.2°C **and** the higher-probability foot exceeds 60% Diabetic probability
- This prevents false positives from borderline or naturally asymmetric healthy feet

### 5. Asymmetry Analysis
- Compares temperature distributions between left and right feet
- Flips the right foot horizontally for pixel-level alignment
- Significance requires **both**: mean inter-foot difference >2.2°C and mean pixel asymmetry >1.0°C

---

## Dataset

| Group | Subjects | Label |
|-------|----------|-------|
| Control Group | 45 | 0 (Healthy) |
| DM Group | 122 | 1 (Diabetic) |

Each subject has:
- Left foot thermal image (PNG) and temperature matrix (CSV)
- Right foot thermal image (PNG) and temperature matrix (CSV)

---

## Step-by-Step Process

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

If PyTorch installation fails, install it separately first:
```bash
pip install torch torchvision
pip install ultralytics>=8.3.0
```

### Step 2: Train Both Models

Both `best_yolo_model.pt` (images) and `best_sklearn_model.joblib` (temperature CSVs) are required for the combined endpoint. The `/predict/patient/images` endpoint works with YOLO alone.

**Option A — Google Colab (recommended, ~15–20 min on free GPU):**

Open `notebooks/train_model_colab.ipynb` in [Google Colab](https://colab.research.google.com):
- Runtime → Change runtime type → T4 GPU → Save
- Runtime → Run all
- The notebook trains **YOLOv11** (image model) then **SVM** (temperature model)
- Download both `best_yolo_model.pt` and `best_sklearn_model.joblib` and place them in `checkpoints/`

**Option B — Local training (~3 hrs on CPU):**

```bash
jupyter notebook notebooks/train_model.ipynb
```

Run all cells. The notebook trains and saves:

| File | Model | Input |
|------|-------|-------|
| `checkpoints/best_yolo_model.pt` | YOLOv11 large | Thermal images |
| `checkpoints/best_sklearn_model.joblib` | SVM RBF | Temperature CSVs |
| `checkpoints/best_model.pth` | CNN (legacy) | Thermal images |

### Step 3: Start the API Server

```bash
uvicorn api.main:app --reload
```

The server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

> The API automatically loads the YOLOv11 model (`best_yolo_model.pt`) on startup.
> If it is not found it falls back to `best_model.pth` (CNN).

### Step 4: Use the API for Predictions

#### Single Foot Endpoints
| Endpoint | Method | Input | Model used |
|----------|--------|-------|------------|
| `/predict/image` | POST | Image file | YOLOv11 (or CNN) |
| `/predict/csv` | POST | CSV file | sklearn |
| `/predict/temperature` | POST | JSON body | sklearn |
| `/predict/batch` | POST | Multiple images | YOLOv11 (or CNN) |

#### Dual-Foot Patient Endpoints
| Endpoint | Method | Input | Models used |
|----------|--------|-------|-------------|
| `/predict/patient/combined` | POST | 2 images + 2 CSVs | **YOLOv11 + sklearn (fused)** ← recommended |
| `/predict/patient/images` | POST | 2 image files | YOLOv11 only |
| `/predict/patient/csv` | POST | 2 CSV files | sklearn only |
| `/predict/patient/temperature` | POST | JSON body | sklearn only |

---

## How to Test the AI

### Option 1: Run the Test Script

```bash
python test_models.py
```

This runs three tests:
1. **YOLOv11 model** — classifies Control and Diabetic image samples
2. **sklearn model** — classifies Control and Diabetic CSV samples
3. **Batch accuracy check** — tests 20 samples and reports accuracy

### Option 2: Using the Swagger UI (Easiest)

1. Start the API server: `uvicorn api.main:app --reload`
2. Open `http://localhost:8000/docs` in your browser
3. Click on any endpoint, then click "Try it out"
4. Upload your files or paste JSON data
5. Click "Execute" to see results

### Option 3: Using curl (Terminal)

**Test with a single thermal image:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@data/Control Group/CG001_M/CG001_M_L.png"
```

**Test with a single CSV file:**
```bash
curl -X POST "http://localhost:8000/predict/csv" \
  -F "file=@data/Control Group/CG001_M/CG001_M_L.csv"
```

**Test combined dual-foot diagnosis (image + CSV — recommended):**
```bash
curl -X POST "http://localhost:8000/predict/patient/combined" \
  -F "left_foot_image=@data/Control Group/CG001_M/CG001_M_L.png" \
  -F "right_foot_image=@data/Control Group/CG001_M/CG001_M_R.png" \
  -F "left_foot_csv=@data/Control Group/CG001_M/CG001_M_L.csv" \
  -F "right_foot_csv=@data/Control Group/CG001_M/CG001_M_R.csv"
```

**Test dual-foot diagnosis with images only:**
```bash
curl -X POST "http://localhost:8000/predict/patient/images" \
  -F "left_foot=@data/Control Group/CG001_M/CG001_M_L.png" \
  -F "right_foot=@data/Control Group/CG001_M/CG001_M_R.png"
```

**Test dual-foot diagnosis with CSVs:**
```bash
curl -X POST "http://localhost:8000/predict/patient/csv" \
  -F "left_foot=@data/Control Group/CG001_M/CG001_M_L.csv" \
  -F "right_foot=@data/Control Group/CG001_M/CG001_M_R.csv"
```

**Test dual-foot diagnosis with JSON temperature values:**
```bash
curl -X POST "http://localhost:8000/predict/patient/temperature" \
  -H "Content-Type: application/json" \
  -d '{
    "left_foot": [[25.5, 26.0, 26.2], [25.8, 26.1, 26.3]],
    "right_foot": [[25.3, 25.9, 26.0], [25.6, 26.0, 26.1]]
  }'
```

### Option 4: Using Python

```python
import requests

# Combined diagnosis — image + CSV per foot (recommended)
url = "http://localhost:8000/predict/patient/combined"
files = {
    "left_foot_image":  open("data/Control Group/CG001_M/CG001_M_L.png", "rb"),
    "right_foot_image": open("data/Control Group/CG001_M/CG001_M_R.png", "rb"),
    "left_foot_csv":    open("data/Control Group/CG001_M/CG001_M_L.csv", "rb"),
    "right_foot_csv":   open("data/Control Group/CG001_M/CG001_M_R.csv", "rb"),
}
response = requests.post(url, files=files)
result = response.json()
print(result["combined_prediction"])   # "Diabetic" or "Control"
print(result["combined_confidence"])   # fused confidence %
print(result["left_foot"]["fusion_method"])  # shows weights used
```

### Option 5: Test from mobile app / same WiFi

Find your PC's local IP:
```powershell
ipconfig
# Look for IPv4 Address e.g. 192.168.1.5
```

From your phone browser (same WiFi): `http://192.168.1.5:8000/docs`

From your mobile app code (React Native):
```javascript
const formData = new FormData()
formData.append('left_foot', { uri: leftFootUri, type: 'image/png', name: 'left.png' })
formData.append('right_foot', { uri: rightFootUri, type: 'image/png', name: 'right.png' })

const response = await fetch('http://192.168.1.5:8000/predict/patient/images', {
    method: 'POST',
    body: formData,
})
const result = await response.json()
// result.is_diabetic      → true/false
// result.combined_prediction → "Diabetic" or "Control"
// result.combined_confidence → confidence percentage
// result.diagnosis_factors   → explanation array
```

### Option 6: Check API Health

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "image_model_loaded": true,
  "image_model_type": "YOLO",
  "sklearn_model_loaded": false
}
```

---

## Example API Response (Combined Dual-Foot Diagnosis)

Response from `POST /predict/patient/combined` — each foot result shows the fused probability alongside individual model contributions.

```json
{
  "success": true,
  "combined_prediction": "Diabetic",
  "combined_confidence": 84.6,
  "is_diabetic": true,
  "left_foot": {
    "prediction": "Diabetic",
    "confidence": 83.1,
    "is_diabetic": true,
    "probabilities": { "Control": 16.9, "Diabetic": 83.1 },
    "yolo_probabilities": { "Control": 12.8, "Diabetic": 87.2 },
    "sklearn_probabilities": { "Control": 23.0, "Diabetic": 77.0 },
    "fusion_method": "weighted_average(yolo=60%, sklearn=40%)"
  },
  "right_foot": {
    "prediction": "Diabetic",
    "confidence": 86.1,
    "is_diabetic": true,
    "probabilities": { "Control": 13.9, "Diabetic": 86.1 },
    "yolo_probabilities": { "Control": 8.2, "Diabetic": 91.8 },
    "sklearn_probabilities": { "Control": 22.5, "Diabetic": 77.5 },
    "fusion_method": "weighted_average(yolo=60%, sklearn=40%)"
  },
  "asymmetry": {
    "mean_asymmetry": 1.2,
    "max_asymmetry": 3.5,
    "left_foot_mean_temp": 28.5,
    "right_foot_mean_temp": 27.3,
    "mean_temp_difference": 1.2,
    "asymmetry_significant": false,
    "threshold_used": 2.2
  },
  "diagnosis_factors": ["Both feet show diabetic indicators"]
}
```

---

## Git Workflow

### Commit and push changes

```bash
# 1. Check what changed
git status
git diff

# 2. Stage only source files — never stage checkpoints or data
git add models/model.py models/data_loader.py api/inference.py api/main.py

# 3. Commit with a descriptive message
git commit -m "your message here"

# 4. Push to GitHub
git push origin main
```

> **Never commit** `checkpoints/` (model weights) or `data/` (thermogram dataset) — they are large binary files already covered by `.gitignore`.

---

## Project Structure

```
transistors-thermal-ai-testing/
├── data/                          # Dataset
│   ├── Control Group/             # 45 healthy subjects
│   │   └── CG001_M/              # Each subject has L/R foot PNG + CSV
│   └── DM Group/                  # 122 diabetic subjects
│       └── DM001_M/
├── models/                        # Model code
│   ├── data_loader.py             # Dataset loading, transforms, YOLO dataset prep
│   ├── preprocessing.py           # Feature extraction and normalization
│   ├── model.py                   # YOLOv11, CNN, ResNet, and sklearn architectures
│   └── trainer.py                 # YOLOTrainer, CNNTrainer, SklearnTrainer
├── api/                           # REST API
│   ├── main.py                    # FastAPI endpoints
│   └── inference.py               # Model loading and prediction logic
├── notebooks/
│   └── train_model.ipynb          # Training notebook
├── checkpoints/                   # Saved models (after training)
│   ├── best_yolo_model.pt         # YOLOv11 best weights (primary)
│   ├── best_model.pth             # CNN best weights (legacy)
│   ├── best_sklearn_model.joblib  # Best sklearn model
│   └── yolo11_dpn/                # Full YOLOv11 training run output
├── test_models.py                 # Model validation tests
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```
