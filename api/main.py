"""
FastAPI Application for DPN Classification
Provides REST API endpoints for thermogram-based diabetic neuropathy detection
"""

import base64
import io
import sys
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.inference import DPNClassifier, get_classifier, predict_patient, calculate_asymmetry


# ==================== App Configuration ====================

app = FastAPI(
    title="DPN Classification API",
    description="""
    API for Diabetic Peripheral Neuropathy (DPN) detection from plantar thermograms.

    Upload a thermogram image OR temperature values and get a classification result
    indicating whether the patient shows signs of diabetic neuropathy.

    ## Input Options
    1. **Thermal Image**: Upload PNG/JPG thermogram image → `/predict/image`
    2. **Temperature CSV**: Upload CSV file with temperature matrix → `/predict/csv`
    3. **Temperature JSON**: Send temperature values as JSON array → `/predict/temperature`

    ## Features
    - Binary classification: Control vs Diabetic
    - Confidence scores and probabilities
    - Multiple input format support
    """,
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Response Models ====================

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    success: bool
    prediction: str
    class_index: int
    confidence: float
    is_diabetic: bool
    probabilities: dict

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": "Diabetic",
                "class_index": 1,
                "confidence": 87.5,
                "is_diabetic": True,
                "probabilities": {
                    "Control": 12.5,
                    "Diabetic": 87.5
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    image_model_loaded: bool
    image_model_type: str
    sklearn_model_loaded: bool
    fusion_model_loaded: bool


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool
    error: str
    detail: Optional[str] = None


class TemperatureInput(BaseModel):
    """Input model for temperature values."""
    temperatures: List[List[float]] = Field(
        ...,
        description="2D array of temperature values (168x65 matrix recommended)",
        example=[[25.5, 26.0, 26.2], [25.8, 26.1, 26.3]]
    )


class DualFootTemperatureInput(BaseModel):
    """Input model for both feet temperature values."""
    left_foot: List[List[float]] = Field(
        ...,
        description="2D array of temperature values for LEFT foot"
    )
    right_foot: List[List[float]] = Field(
        ...,
        description="2D array of temperature values for RIGHT foot"
    )


class FootResult(BaseModel):
    """Result for a single foot. Includes fusion details when both image + CSV were used."""
    prediction: str
    confidence: float
    is_diabetic: bool
    probabilities: dict
    # Present only when YOLOv11 + sklearn fusion was applied
    yolo_probabilities: Optional[dict] = None
    sklearn_probabilities: Optional[dict] = None
    fusion_method: Optional[str] = None


class AsymmetryResult(BaseModel):
    """Asymmetry analysis between feet."""
    mean_asymmetry: float
    max_asymmetry: float
    left_foot_mean_temp: float
    right_foot_mean_temp: float
    mean_temp_difference: float
    asymmetry_significant: bool
    threshold_used: float


class PatientPredictionResponse(BaseModel):
    """Response model for dual-foot patient prediction."""
    success: bool
    combined_prediction: str
    combined_confidence: float
    is_diabetic: bool
    left_foot: Optional[FootResult] = None
    right_foot: Optional[FootResult] = None
    asymmetry: Optional[AsymmetryResult] = None
    diagnosis_factors: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "combined_prediction": "Diabetic",
                "combined_confidence": 89.5,
                "is_diabetic": True,
                "left_foot": {
                    "prediction": "Diabetic",
                    "confidence": 87.2,
                    "is_diabetic": True,
                    "probabilities": {"Control": 12.8, "Diabetic": 87.2}
                },
                "right_foot": {
                    "prediction": "Diabetic",
                    "confidence": 91.8,
                    "is_diabetic": True,
                    "probabilities": {"Control": 8.2, "Diabetic": 91.8}
                },
                "asymmetry": {
                    "mean_asymmetry": 1.2,
                    "max_asymmetry": 3.5,
                    "left_foot_mean_temp": 28.5,
                    "right_foot_mean_temp": 27.3,
                    "mean_temp_difference": 1.2,
                    "asymmetry_significant": False,
                    "threshold_used": 2.2
                },
                "diagnosis_factors": ["Both feet show diabetic indicators"]
            }
        }


# ==================== Global State ====================

# image_classifier holds whichever image model is available: YOLOv11 (preferred) or CNN (fallback)
image_classifier: Optional[DPNClassifier] = None
sklearn_classifier: Optional[DPNClassifier] = None
fusion_model = None  # LogisticRegression meta-classifier (checkpoints/best_fusion_model.joblib)

# Keep a reference to the legacy name so predict_patient() calls still work
cnn_classifier: Optional[DPNClassifier] = None


# ==================== Startup Event ====================

@app.on_event("startup")
async def load_models():
    """Load image model (YOLOv11 preferred, CNN fallback), sklearn model, and fusion meta-classifier on startup."""
    global image_classifier, sklearn_classifier, cnn_classifier, fusion_model

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    # --- Image model: try YOLOv11 first, then CNN ---
    yolo_paths = [
        checkpoints_dir / "best_yolo_model.pt",
        checkpoints_dir / "yolo11_dpn" / "weights" / "best.pt",
    ]
    cnn_path = checkpoints_dir / "best_model.pth"

    loaded_image_model = False
    for yolo_path in yolo_paths:
        if yolo_path.exists():
            try:
                image_classifier = DPNClassifier(
                    model_path=str(yolo_path),
                    model_type="yolo"
                )
                cnn_classifier = image_classifier  # alias for predict_patient()
                print(f"YOLOv11 model loaded from {yolo_path}")
                loaded_image_model = True
                break
            except Exception as e:
                print(f"Error loading YOLOv11 model from {yolo_path}: {e}")

    if not loaded_image_model and cnn_path.exists():
        try:
            image_classifier = DPNClassifier(
                model_path=str(cnn_path),
                model_type="cnn"
            )
            cnn_classifier = image_classifier  # alias for predict_patient()
            print(f"CNN model loaded from {cnn_path} (YOLOv11 not found, using fallback)")
            loaded_image_model = True
        except Exception as e:
            print(f"Error loading CNN model: {e}")

    if not loaded_image_model:
        print("Warning: No image model found. Run training first.")
        print(f"  Expected YOLO: {yolo_paths[0]}")
        print(f"  Expected CNN:  {cnn_path}")

    # --- sklearn model (for temperature CSV/JSON endpoints) ---
    sklearn_path = checkpoints_dir / "best_sklearn_model.joblib"
    if sklearn_path.exists():
        try:
            sklearn_classifier = DPNClassifier(
                model_path=str(sklearn_path),
                model_type="sklearn"
            )
            print(f"sklearn model loaded from {sklearn_path}")
        except Exception as e:
            print(f"Error loading sklearn model: {e}")
    else:
        print(f"Warning: sklearn model not found at {sklearn_path}")

    # --- Fusion meta-classifier (optional but recommended for FLIR Lepton deployment) ---
    fusion_path = checkpoints_dir / "best_fusion_model.joblib"
    if fusion_path.exists():
        try:
            import joblib as _joblib
            fusion_model = _joblib.load(fusion_path)
            print(f"Fusion meta-classifier loaded from {fusion_path}")
        except Exception as e:
            print(f"Warning: could not load fusion model: {e}")
    else:
        print(f"Info: No fusion model at {fusion_path} — using fixed 60/40 weights.")
        print("      Run python train_fusion.py to train it.")

    if image_classifier is None and sklearn_classifier is None:
        print("WARNING: No models loaded! Run the training notebook first.")


# ==================== Endpoints ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DPN Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    model_type = image_classifier.model_type.upper() if image_classifier else "none"
    return HealthResponse(
        status="healthy",
        image_model_loaded=image_classifier is not None,
        image_model_type=model_type,
        sklearn_model_loaded=sklearn_classifier is not None,
        fusion_model_loaded=fusion_model is not None,
    )


@app.post(
    "/predict/image",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Prediction"]
)
async def predict_image(
    file: UploadFile = File(..., description="Thermogram image file (PNG, JPG, JPEG)")
):
    """
    Classify a plantar thermogram IMAGE.

    Upload a thermal camera image and receive a prediction indicating whether
    the patient shows signs of diabetic peripheral neuropathy.

    - **file**: Thermogram image (PNG, JPG, or JPEG format)

    Returns classification result with confidence scores.
    """
    if image_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not loaded. Please train and save a YOLOv11 or CNN model first."
        )

    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: PNG, JPG, JPEG"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = image_classifier.predict(image, return_proba=True)

        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            class_index=result["class_index"],
            confidence=result["confidence"],
            is_diabetic=result["is_diabetic"],
            probabilities=result["probabilities"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/csv",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Prediction"]
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV file with temperature matrix")
):
    """
    Classify using temperature VALUES from CSV file.

    Upload a CSV file containing temperature readings from a thermal camera.
    The CSV should contain a matrix of temperature values (no headers).

    - **file**: CSV file with temperature matrix (recommended: 168 rows x 65 columns)

    Returns classification result with confidence scores.
    """
    if sklearn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="sklearn model not loaded. Please ensure best_sklearn_model.joblib exists in checkpoints/"
        )

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file."
        )

    try:
        contents = await file.read()
        # Parse CSV from bytes
        temp_file = io.StringIO(contents.decode('utf-8'))
        data = pd.read_csv(temp_file, header=None).values.astype(np.float32)

        # Save temporarily for the classifier
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(data).to_csv(f.name, header=False, index=False)
            temp_path = f.name

        result = sklearn_classifier.predict(temp_path, return_proba=True)

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            class_index=result["class_index"],
            confidence=result.get("confidence", 0.0),
            is_diabetic=result["is_diabetic"],
            probabilities=result.get("probabilities", {"Control": 0, "Diabetic": 0})
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/temperature",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Prediction"]
)
async def predict_temperature(
    data: TemperatureInput = Body(..., description="Temperature values as 2D JSON array")
):
    """
    Classify using temperature VALUES from JSON.

    Send temperature readings directly as a JSON array. Useful for real-time
    integration with thermal cameras that output temperature data.

    - **temperatures**: 2D array of temperature values (recommended: 168x65 matrix)

    Example request body:
    ```json
    {
        "temperatures": [
            [25.5, 26.0, 26.2, ...],
            [25.8, 26.1, 26.3, ...],
            ...
        ]
    }
    ```

    Returns classification result with confidence scores.
    """
    if sklearn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="sklearn model not loaded. Please ensure best_sklearn_model.joblib exists in checkpoints/"
        )

    try:
        # Convert to numpy array
        temp_matrix = np.array(data.temperatures, dtype=np.float32)

        if temp_matrix.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail="Temperature data must be a 2D array"
            )

        # Save temporarily for the classifier
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(temp_matrix).to_csv(f.name, header=False, index=False)
            temp_path = f.name

        result = sklearn_classifier.predict(temp_path, return_proba=True)

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            class_index=result["class_index"],
            confidence=result.get("confidence", 0.0),
            is_diabetic=result["is_diabetic"],
            probabilities=result.get("probabilities", {"Control": 0, "Diabetic": 0})
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple thermogram images")
):
    """
    Classify multiple thermogram images in batch.

    Upload multiple images and receive predictions for each.
    """
    if image_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not loaded. Please train and save a YOLOv11 or CNN model first."
        )

    results = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = image_classifier.predict(image, return_proba=True)
            results.append({
                "filename": file.filename,
                "success": True,
                **result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "total": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }


# ==================== Dual-Foot Patient Endpoints ====================

@app.post(
    "/predict/patient/images",
    response_model=PatientPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Patient Diagnosis"]
)
async def predict_patient_images(
    left_foot: UploadFile = File(..., description="Left foot thermogram image"),
    right_foot: UploadFile = File(..., description="Right foot thermogram image")
):
    """
    Classify a patient using BOTH feet thermal images.

    This endpoint provides clinically-accurate DPN diagnosis by:
    1. Analyzing each foot individually
    2. Combining results for final diagnosis

    Upload thermal images for both left and right feet.
    """
    if image_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not loaded. Please train and save a YOLOv11 or CNN model first."
        )

    allowed_types = ["image/png", "image/jpeg", "image/jpg"]

    for file, foot_name in [(left_foot, "left"), (right_foot, "right")]:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {foot_name} foot: {file.content_type}. Allowed: PNG, JPG, JPEG"
            )

    try:
        # Load images
        left_contents = await left_foot.read()
        right_contents = await right_foot.read()

        left_image = Image.open(io.BytesIO(left_contents)).convert("RGB")
        right_image = Image.open(io.BytesIO(right_contents)).convert("RGB")

        # Get predictions using predict_patient
        result = predict_patient(
            cnn_classifier=image_classifier,
            sklearn_classifier=sklearn_classifier,
            left_image=left_image,
            right_image=right_image,
            fusion_model=fusion_model,
        )

        return PatientPredictionResponse(
            success=True,
            combined_prediction=result["combined_prediction"],
            combined_confidence=result["combined_confidence"],
            is_diabetic=result["is_diabetic"],
            left_foot=FootResult(**{k: v for k, v in result["left_foot"].items() if k != "input_type"}) if result["left_foot"] else None,
            right_foot=FootResult(**{k: v for k, v in result["right_foot"].items() if k != "input_type"}) if result["right_foot"] else None,
            asymmetry=AsymmetryResult(**result["asymmetry"]) if result["asymmetry"] else None,
            diagnosis_factors=result["diagnosis_factors"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/patient/csv",
    response_model=PatientPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Patient Diagnosis"]
)
async def predict_patient_csv(
    left_foot: UploadFile = File(..., description="Left foot temperature CSV"),
    right_foot: UploadFile = File(..., description="Right foot temperature CSV")
):
    """
    Classify a patient using BOTH feet temperature CSV files.

    This endpoint provides clinically-accurate DPN diagnosis by:
    1. Analyzing each foot individually
    2. Calculating temperature asymmetry between feet
    3. Combining results for final diagnosis

    Upload CSV files with temperature matrices for both feet.
    """
    if sklearn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="sklearn model not loaded. Please ensure best_sklearn_model.joblib exists in checkpoints/"
        )

    for file, foot_name in [(left_foot, "left"), (right_foot, "right")]:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {foot_name} foot. Please upload a CSV file."
            )

    try:
        import tempfile

        # Read and save CSV files temporarily
        left_contents = await left_foot.read()
        right_contents = await right_foot.read()

        left_data = pd.read_csv(io.StringIO(left_contents.decode('utf-8')), header=None).values.astype(np.float32)
        right_data = pd.read_csv(io.StringIO(right_contents.decode('utf-8')), header=None).values.astype(np.float32)

        # Save to temp files for classifier
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(left_data).to_csv(f.name, header=False, index=False)
            left_temp_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(right_data).to_csv(f.name, header=False, index=False)
            right_temp_path = f.name

        # Get predictions using predict_patient
        result = predict_patient(
            cnn_classifier=image_classifier,
            sklearn_classifier=sklearn_classifier,
            left_csv=left_temp_path,
            right_csv=right_temp_path,
            left_temps=left_data,
            right_temps=right_data,
            fusion_model=fusion_model,
        )

        # Cleanup temp files
        Path(left_temp_path).unlink(missing_ok=True)
        Path(right_temp_path).unlink(missing_ok=True)

        return PatientPredictionResponse(
            success=True,
            combined_prediction=result["combined_prediction"],
            combined_confidence=result["combined_confidence"],
            is_diabetic=result["is_diabetic"],
            left_foot=FootResult(**{k: v for k, v in result["left_foot"].items() if k != "input_type"}) if result["left_foot"] else None,
            right_foot=FootResult(**{k: v for k, v in result["right_foot"].items() if k != "input_type"}) if result["right_foot"] else None,
            asymmetry=AsymmetryResult(**result["asymmetry"]) if result["asymmetry"] else None,
            diagnosis_factors=result["diagnosis_factors"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/patient/temperature",
    response_model=PatientPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Patient Diagnosis"]
)
async def predict_patient_temperature(
    data: DualFootTemperatureInput = Body(..., description="Temperature values for both feet as JSON")
):
    """
    Classify a patient using BOTH feet temperature values from JSON.

    This endpoint provides clinically-accurate DPN diagnosis by:
    1. Analyzing each foot individually
    2. Calculating temperature asymmetry between feet
    3. Combining results for final diagnosis

    Send temperature readings for both feet as JSON arrays.

    Example request body:
    ```json
    {
        "left_foot": [[25.5, 26.0, ...], [25.8, 26.1, ...], ...],
        "right_foot": [[25.3, 25.9, ...], [25.6, 26.0, ...], ...]
    }
    ```
    """
    if sklearn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="sklearn model not loaded. Please ensure best_sklearn_model.joblib exists in checkpoints/"
        )

    try:
        import tempfile

        # Convert to numpy arrays
        left_temps = np.array(data.left_foot, dtype=np.float32)
        right_temps = np.array(data.right_foot, dtype=np.float32)

        for temps, foot_name in [(left_temps, "left"), (right_temps, "right")]:
            if temps.ndim != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Temperature data for {foot_name} foot must be a 2D array"
                )

        # Save to temp files for classifier
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(left_temps).to_csv(f.name, header=False, index=False)
            left_temp_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(right_temps).to_csv(f.name, header=False, index=False)
            right_temp_path = f.name

        # Get predictions using predict_patient
        result = predict_patient(
            cnn_classifier=image_classifier,
            sklearn_classifier=sklearn_classifier,
            left_csv=left_temp_path,
            right_csv=right_temp_path,
            left_temps=left_temps,
            right_temps=right_temps,
            fusion_model=fusion_model,
        )

        # Cleanup temp files
        Path(left_temp_path).unlink(missing_ok=True)
        Path(right_temp_path).unlink(missing_ok=True)

        return PatientPredictionResponse(
            success=True,
            combined_prediction=result["combined_prediction"],
            combined_confidence=result["combined_confidence"],
            is_diabetic=result["is_diabetic"],
            left_foot=FootResult(**{k: v for k, v in result["left_foot"].items() if k != "input_type"}) if result["left_foot"] else None,
            right_foot=FootResult(**{k: v for k, v in result["right_foot"].items() if k != "input_type"}) if result["right_foot"] else None,
            asymmetry=AsymmetryResult(**result["asymmetry"]) if result["asymmetry"] else None,
            diagnosis_factors=result["diagnosis_factors"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/patient/combined",
    response_model=PatientPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Patient Diagnosis"]
)
async def predict_patient_combined(
    left_foot_image: UploadFile = File(..., description="Left foot thermogram image (PNG/JPG)"),
    right_foot_image: UploadFile = File(..., description="Right foot thermogram image (PNG/JPG)"),
    left_foot_csv: UploadFile = File(..., description="Left foot temperature CSV"),
    right_foot_csv: UploadFile = File(..., description="Right foot temperature CSV"),
):
    """
    Combined DPN diagnosis using BOTH thermogram images (YOLOv11) AND
    temperature CSV data (sklearn) for each foot.

    Per-foot prediction is a weighted fusion:
      - YOLOv11 image model: 60% weight (primary — 97.5% accuracy)
      - sklearn temperature model: 40% weight (secondary — raw temperature signal)

    Upload all four files: two foot images + two temperature CSVs.
    The response includes per-modality probabilities alongside the fused result.
    """
    if image_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not loaded. Please train and save a YOLOv11 model first."
        )
    if sklearn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="sklearn model not loaded. Please ensure best_sklearn_model.joblib exists in checkpoints/"
        )

    allowed_image_types = {"image/png", "image/jpeg", "image/jpg"}
    for file, name in [(left_foot_image, "left_foot_image"), (right_foot_image, "right_foot_image")]:
        if file.content_type not in allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {name}: {file.content_type}. Allowed: PNG, JPG, JPEG"
            )
    for file, name in [(left_foot_csv, "left_foot_csv"), (right_foot_csv, "right_foot_csv")]:
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {name}. Please upload a CSV file."
            )

    try:
        import tempfile

        # Read images
        left_img  = Image.open(io.BytesIO(await left_foot_image.read())).convert("RGB")
        right_img = Image.open(io.BytesIO(await right_foot_image.read())).convert("RGB")

        # Read CSVs → numpy + temp files on disk for sklearn classifier
        left_csv_bytes  = await left_foot_csv.read()
        right_csv_bytes = await right_foot_csv.read()

        left_temps  = pd.read_csv(io.StringIO(left_csv_bytes.decode("utf-8")),  header=None).values.astype(np.float32)
        right_temps = pd.read_csv(io.StringIO(right_csv_bytes.decode("utf-8")), header=None).values.astype(np.float32)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(left_temps).to_csv(f.name, header=False, index=False)
            left_csv_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame(right_temps).to_csv(f.name, header=False, index=False)
            right_csv_path = f.name

        result = predict_patient(
            cnn_classifier=image_classifier,
            sklearn_classifier=sklearn_classifier,
            left_image=left_img,
            right_image=right_img,
            left_csv=left_csv_path,
            right_csv=right_csv_path,
            left_temps=left_temps,
            right_temps=right_temps,
        )

        Path(left_csv_path).unlink(missing_ok=True)
        Path(right_csv_path).unlink(missing_ok=True)

        def _foot_result(foot_data):
            if foot_data is None:
                return None
            allowed = {
                "prediction", "confidence", "is_diabetic", "probabilities",
                "yolo_probabilities", "sklearn_probabilities", "fusion_method",
            }
            return FootResult(**{k: v for k, v in foot_data.items() if k in allowed})

        return PatientPredictionResponse(
            success=True,
            combined_prediction=result["combined_prediction"],
            combined_confidence=result["combined_confidence"],
            is_diabetic=result["is_diabetic"],
            left_foot=_foot_result(result["left_foot"]),
            right_foot=_foot_result(result["right_foot"]),
            asymmetry=AsymmetryResult(**result["asymmetry"]) if result["asymmetry"] else None,
            diagnosis_factors=result["diagnosis_factors"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ==================== Mobile / FLIR Lepton Endpoint ====================

class MobilePatientInput(BaseModel):
    """
    JSON input for the FLIR Lepton 3.5 mobile app.

    The camera SDK gives you raw pixel bytes (image) and a temperature matrix.
    Encode the rendered thermal image as base64 PNG, and send the temperature
    matrix as a 2-D array of floats.  Both fields are required for full
    dual-modality fusion (YOLO + sklearn + meta-classifier).

    Fields
    ------
    left_image_b64   : base64-encoded PNG of the left foot thermal image
    right_image_b64  : base64-encoded PNG of the right foot thermal image
    left_temperatures: 2-D array of temperature values for the left foot
    right_temperatures: 2-D array of temperature values for the right foot
    """
    left_image_b64: str = Field(..., description="Base64-encoded PNG of left foot thermal image")
    right_image_b64: str = Field(..., description="Base64-encoded PNG of right foot thermal image")
    left_temperatures: List[List[float]] = Field(..., description="2-D temperature matrix for left foot (e.g. 120x160)")
    right_temperatures: List[List[float]] = Field(..., description="2-D temperature matrix for right foot")


@app.post(
    "/predict/patient/mobile",
    response_model=PatientPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Patient Diagnosis"],
    summary="FLIR Lepton 3.5 — combined image + temperature (JSON)",
)
async def predict_patient_mobile(data: MobilePatientInput):
    """
    Recommended endpoint for the FLIR Lepton 3.5 mobile app.

    Accepts a single JSON body containing:
    - Base64-encoded thermal images for both feet
    - Raw temperature matrices for both feet

    Runs full dual-modality fusion:
      YOLOv11 (image) + sklearn SVM (temperature) → meta-classifier → final verdict

    **How to call from the mobile app:**
    ```json
    POST /predict/patient/mobile
    Content-Type: application/json

    {
      "left_image_b64": "<base64 PNG>",
      "right_image_b64": "<base64 PNG>",
      "left_temperatures": [[28.5, 28.6, ...], ...],
      "right_temperatures": [[28.3, 28.4, ...], ...]
    }
    ```
    """
    if image_classifier is None:
        raise HTTPException(status_code=503, detail="Image model not loaded.")
    if sklearn_classifier is None:
        raise HTTPException(status_code=503, detail="sklearn model not loaded.")

    try:
        import tempfile

        # Decode base64 images
        try:
            left_img_bytes  = base64.b64decode(data.left_image_b64)
            right_img_bytes = base64.b64decode(data.right_image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")

        left_img  = Image.open(io.BytesIO(left_img_bytes)).convert("RGB")
        right_img = Image.open(io.BytesIO(right_img_bytes)).convert("RGB")

        # Temperature matrices
        left_temps  = np.array(data.left_temperatures,  dtype=np.float32)
        right_temps = np.array(data.right_temperatures, dtype=np.float32)

        for arr, name in [(left_temps, "left"), (right_temps, "right")]:
            if arr.ndim != 2:
                raise HTTPException(status_code=400, detail=f"{name}_temperatures must be a 2-D array.")

        # Write temps to temp CSV files (sklearn classifier reads from file path)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(left_temps).to_csv(f.name, header=False, index=False)
            left_csv_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(right_temps).to_csv(f.name, header=False, index=False)
            right_csv_path = f.name

        result = predict_patient(
            cnn_classifier=image_classifier,
            sklearn_classifier=sklearn_classifier,
            left_image=left_img,
            right_image=right_img,
            left_csv=left_csv_path,
            right_csv=right_csv_path,
            left_temps=left_temps,
            right_temps=right_temps,
            fusion_model=fusion_model,
        )

        Path(left_csv_path).unlink(missing_ok=True)
        Path(right_csv_path).unlink(missing_ok=True)

        def _foot_result(foot_data):
            if foot_data is None:
                return None
            allowed = {
                "prediction", "confidence", "is_diabetic", "probabilities",
                "yolo_probabilities", "sklearn_probabilities", "fusion_method",
            }
            return FootResult(**{k: v for k, v in foot_data.items() if k in allowed})

        return PatientPredictionResponse(
            success=True,
            combined_prediction=result["combined_prediction"],
            combined_confidence=result["combined_confidence"],
            is_diabetic=result["is_diabetic"],
            left_foot=_foot_result(result["left_foot"]),
            right_foot=_foot_result(result["right_foot"]),
            asymmetry=AsymmetryResult(**result["asymmetry"]) if result["asymmetry"] else None,
            diagnosis_factors=result["diagnosis_factors"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
