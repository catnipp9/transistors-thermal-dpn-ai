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

## AI Models Used

### 1. Convolutional Neural Network (CNN) - For Thermal Images
- **Framework**: PyTorch
- **Architecture**: LightweightCNN (custom-built)
  - 3 convolutional blocks with batch normalization and ReLU activation
  - Adaptive average pooling for flexible input sizes
  - Fully connected classifier with dropout regularization
- **Input**: RGB thermal images (PNG/JPG)
- **Output**: Binary classification (Control vs Diabetic) with confidence percentage

### 2. Classical Machine Learning Models - For Temperature Values
- **Framework**: scikit-learn
- **Models trained and compared**:
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
  - Logistic Regression
- **Input**: CSV temperature matrices (168x65) flattened into feature vectors
- **Output**: Binary classification (Control vs Diabetic) with confidence percentage
- The best performing model is automatically saved after training

### 3. Asymmetry Analysis
- Compares temperature distributions between left and right feet
- Flips the right foot horizontally for pixel-level alignment
- Calculates mean, max, and standard deviation of temperature differences
- Clinical threshold: >2.2 degrees Celsius flags significant asymmetry

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

If PyTorch installation fails, install it separately:
```bash
pip install torch torchvision
```

### Step 2: Train the Models

Open and run the training notebook:

```bash
jupyter notebook notebooks/train_model.ipynb
```

Run all cells in order. The notebook will:
1. Load all thermogram data from the `data/` directory
2. Split data into training (70%), validation (15%), and test (15%) sets
3. Train the CNN model on thermal images
4. Train multiple sklearn models on temperature values
5. Compare all models and save the best ones to `checkpoints/`

After training completes, you will see:
- `checkpoints/best_model.pth` - Best CNN model
- `checkpoints/best_sklearn_model.joblib` - Best sklearn model
- `checkpoints/training_history.png` - Training loss/accuracy curves
- `checkpoints/model_comparison_results.csv` - Comparison table of all models

### Step 3: Start the API Server

```bash
uvicorn api.main:app --reload
```

The server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Step 4: Use the API for Predictions

The API provides the following endpoints:

#### Single Foot Endpoints
| Endpoint | Method | Input | Description |
|----------|--------|-------|-------------|
| `/predict/image` | POST | Image file | Classify one thermal image |
| `/predict/csv` | POST | CSV file | Classify one temperature CSV |
| `/predict/temperature` | POST | JSON body | Classify one temperature array |
| `/predict/batch` | POST | Multiple images | Classify multiple images at once |

#### Dual-Foot Patient Endpoints (Recommended)
| Endpoint | Method | Input | Description |
|----------|--------|-------|-------------|
| `/predict/patient/images` | POST | 2 image files | Both feet thermal images |
| `/predict/patient/csv` | POST | 2 CSV files | Both feet temperature CSVs |
| `/predict/patient/temperature` | POST | JSON body | Both feet temperature arrays |

---

## How to Test the AI

### Option 1: Using the Swagger UI (Easiest)

1. Start the API server: `uvicorn api.main:app --reload`
2. Open `http://localhost:8000/docs` in your browser
3. Click on any endpoint, then click "Try it out"
4. Upload your files or paste JSON data
5. Click "Execute" to see results

### Option 2: Using curl (Terminal)

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

**Test dual-foot diagnosis with images:**
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

### Option 3: Using Python

```python
import requests

# Test with thermal images (both feet)
url = "http://localhost:8000/predict/patient/images"
files = {
    "left_foot": open("data/Control Group/CG001_M/CG001_M_L.png", "rb"),
    "right_foot": open("data/Control Group/CG001_M/CG001_M_R.png", "rb"),
}
response = requests.post(url, files=files)
print(response.json())
```

### Option 4: Check API Health

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "cnn_model_loaded": true,
  "sklearn_model_loaded": true
}
```

If either model shows `false`, make sure you ran the training notebook first.

---

## Example API Response (Dual-Foot Diagnosis)

```json
{
  "success": true,
  "combined_prediction": "Diabetic",
  "combined_confidence": 89.5,
  "is_diabetic": true,
  "left_foot": {
    "prediction": "Diabetic",
    "confidence": 87.2,
    "is_diabetic": true,
    "probabilities": { "Control": 12.8, "Diabetic": 87.2 }
  },
  "right_foot": {
    "prediction": "Diabetic",
    "confidence": 91.8,
    "is_diabetic": true,
    "probabilities": { "Control": 8.2, "Diabetic": 91.8 }
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

## Project Structure

```
Thesis_DPN/
├── data/                          # Dataset
│   ├── Control Group/             # 45 healthy subjects
│   │   └── CG001_M/              # Each subject has L/R foot PNG + CSV
│   └── DM Group/                  # 122 diabetic subjects
│       └── DM001_M/
├── models/                        # Model code
│   ├── data_loader.py             # Dataset loading and transforms
│   ├── preprocessing.py           # Feature extraction and normalization
│   ├── model.py                   # CNN and sklearn model architectures
│   └── trainer.py                 # Training utilities
├── api/                           # REST API
│   ├── main.py                    # FastAPI endpoints
│   └── inference.py               # Model loading and prediction logic
├── notebooks/
│   └── train_model.ipynb          # Training notebook
├── checkpoints/                   # Saved models (after training)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```
