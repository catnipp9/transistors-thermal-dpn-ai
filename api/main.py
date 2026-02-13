"""
FastAPI Application for DPN Classification
Provides REST API endpoints for thermogram-based diabetic neuropathy detection
"""

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
    cnn_model_loaded: bool
    sklearn_model_loaded: bool


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
    """Result for a single foot."""
    prediction: str
    confidence: float
    is_diabetic: bool
    probabilities: dict


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

cnn_classifier: Optional[DPNClassifier] = None
sklearn_classifier: Optional[DPNClassifier] = None


# ==================== Startup Event ====================

@app.on_event("startup")
async def load_models():
    """Load both CNN and sklearn models on startup."""
    global cnn_classifier, sklearn_classifier

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    # Load CNN model (for thermal images)
    cnn_path = checkpoints_dir / "best_model.pth"
    if cnn_path.exists():
        try:
            cnn_classifier = DPNClassifier(
                model_path=str(cnn_path),
                model_type="cnn"
            )
            print(f"CNN model loaded from {cnn_path}")
        except Exception as e:
            print(f"Error loading CNN model: {e}")
    else:
        print(f"Warning: CNN model not found at {cnn_path}")

    # Load sklearn model (for temperature values)
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

    if cnn_classifier is None and sklearn_classifier is None:
        print("WARNING: No models loaded! Run training notebook first.")


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
    return HealthResponse(
        status="healthy",
        cnn_model_loaded=cnn_classifier is not None,
        sklearn_model_loaded=sklearn_classifier is not None
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
    if cnn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="CNN model not loaded. Please ensure best_model.pth exists in checkpoints/"
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
        result = cnn_classifier.predict(image, return_proba=True)

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
    if cnn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="CNN model not loaded."
        )

    results = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = cnn_classifier.predict(image, return_proba=True)
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
    if cnn_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="CNN model not loaded. Please ensure best_model.pth exists in checkpoints/"
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
            cnn_classifier=cnn_classifier,
            sklearn_classifier=sklearn_classifier,
            left_image=left_image,
            right_image=right_image
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
            cnn_classifier=cnn_classifier,
            sklearn_classifier=sklearn_classifier,
            left_csv=left_temp_path,
            right_csv=right_temp_path,
            left_temps=left_data,
            right_temps=right_data
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
            cnn_classifier=cnn_classifier,
            sklearn_classifier=sklearn_classifier,
            left_csv=left_temp_path,
            right_csv=right_temp_path,
            left_temps=left_temps,
            right_temps=right_temps
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


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
