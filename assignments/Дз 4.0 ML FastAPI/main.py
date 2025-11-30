"""
FastAPI application for churn prediction
Complete implementation covering all 14 days
"""

import time
import psutil
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import schemas and models
from schemas.churn import (
    FeatureVectorChurn, DatasetRowChurn, PredictionResponseChurn,
    BatchPredictionRequest, BatchPredictionResponse,
    TrainingResponse, ModelStatus, HealthResponse, ErrorResponse
)
from ml.churn_model import ChurnModelPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML service for customer churn prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model pipeline
model_pipeline = ChurnModelPipeline()

# Start time for uptime tracking
start_time = time.time()


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler (Day 10)"""
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred",
        details={"error_details": str(exc)},
        timestamp=datetime.now().isoformat()
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Value error handler"""
    error_response = ErrorResponse(
        error="ValueError",
        message=str(exc),
        timestamp=datetime.now().isoformat()
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.dict()
    )


# Root endpoint (Day 1)
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "ML churn service is running"}


# Health check endpoint (Day 13)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Get system metrics
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024

        # Get model status
        model_status = ModelStatus(**model_pipeline.get_model_status())

        health_response = HealthResponse(
            status="healthy",
            version="1.0.0",
            model_status=model_status,
            uptime_seconds=time.time() - start_time,
            memory_usage_mb=memory_usage_mb,
            timestamp=datetime.now().isoformat()
        )

        return health_response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponseChurn)
async def predict_churn(customer_features: FeatureVectorChurn, customer_id: str = None):
    """
    Predict churn for a single customer (Day 7)
    """
    try:
        # Check if model is trained
        if not model_pipeline.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not trained. Please train the model first using /model/train endpoint"
            )

        # Convert to dictionary
        customer_data = customer_features.dict()

        # Make prediction
        start_time = time.time()
        predictions = model_pipeline.predict([customer_data])
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Create response
        prediction = predictions[0]
        prediction['customer_id'] = customer_id
        prediction['processing_time_ms'] = processing_time

        return PredictionResponseChurn(**prediction)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers (Day 11)
    """
    try:
        # Check if model is trained
        if not model_pipeline.model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not trained. Please train the model first using /model/train endpoint"
            )

        # Convert to list of dictionaries
        customers_data = [customer.dict() for customer in request.customers]

        # Make predictions
        start_time = time.time()
        predictions = model_pipeline.predict(customers_data)
        processing_time = (time.time() - start_time) * 1000

        # Create prediction responses
        prediction_responses = []
        churn_risk_summary = {"low": 0, "medium": 0, "high": 0}
        total_churn_probability = 0.0

        for i, prediction in enumerate(predictions):
            prediction_response = PredictionResponseChurn(**prediction)
            prediction_responses.append(prediction_response)

            # Update statistics
            churn_risk_summary[prediction['risk_level']] += 1
            total_churn_probability += prediction['churn_probability']

        # Calculate averages
        average_churn_probability = total_churn_probability / len(predictions) if predictions else 0.0

        # Create batch response
        batch_response = BatchPredictionResponse(
            predictions=prediction_responses,
            total_customers=len(request.customers),
            churn_risk_summary=churn_risk_summary,
            average_churn_probability=average_churn_probability,
            processing_time_ms=processing_time
        )

        return batch_response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Model management endpoints
@app.post("/model/train", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks, data_path: str = "churn_dataset.csv"):
    """
    Train the churn model (Day 6)
    """
    try:
        # Load data
        df = model_pipeline.load_data(data_path)

        # Preprocess data
        df_processed = model_pipeline.preprocess_data(df)

        # Train model
        training_result = model_pipeline.train_model(df_processed)

        # Create response
        training_response = TrainingResponse(
            message="Model trained successfully",
            model_version=training_result['model_version'],
            training_samples=training_result['training_samples'],
            test_samples=training_result['test_samples'],
            accuracy=training_result['metrics'].get('accuracy'),
            precision=training_result['metrics'].get('precision'),
            recall=training_result['metrics'].get('recall'),
            f1_score=training_result['metrics'].get('f1_score'),
            auc_roc=training_result['metrics'].get('auc_roc'),
            training_time_seconds=training_result['training_time_seconds'],
            feature_importance=training_result['feature_importance']
        )

        return training_response

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file not found: {data_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """
    Get current model status (Day 12)
    """
    try:
        status_data = model_pipeline.get_model_status()
        return ModelStatus(**status_data)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/model/load")
async def load_model():
    """Load a previously trained model"""
    try:
        success = model_pipeline.load_model()
        if success:
            return {"message": "Model loaded successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No trained model found"
            )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Data management endpoints
@app.post("/data/upload")
async def upload_data(background_tasks: BackgroundTasks, file_path: str = "churn_dataset.csv"):
    """
    Upload and validate dataset (Day 3-4)
    """
    try:
        # Try to load and validate the data
        df = model_pipeline.load_data(file_path)

        # Preprocess to check for issues
        df_processed = model_pipeline.preprocess_data(df)

        return {
            "message": "Data uploaded and validated successfully",
            "total_rows": len(df),
            "columns": list(df.columns),
            "churn_distribution": {
                "churned": int(df['churn'].sum()),
                "stayed": int(len(df) - df['churn'].sum())
            },
            "validation_passed": True
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file not found: {file_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/data/stats")
async def get_data_stats(file_path: str = "churn_dataset.csv"):
    """
    Get dataset statistics (Day 4)
    """
    try:
        # Load data
        df = model_pipeline.load_data(file_path)

        # Calculate statistics
        stats = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "numeric_columns": model_pipeline.numerical_columns,
            "categorical_columns": model_pipeline.categorical_columns,
            "churn_distribution": {
                "churned": int(df['churn'].sum()),
                "stayed": int(len(df) - df['churn'].sum()),
                "churn_rate": float(df['churn'].mean())
            }
        }

        # Add numeric column statistics
        for col in model_pipeline.numerical_columns:
            if col in df.columns:
                stats[f"{col}_stats"] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }

        # Add categorical column distributions
        for col in model_pipeline.categorical_columns:
            if col in df.columns:
                stats[f"{col}_distribution"] = df[col].value_counts().to_dict()

        return stats

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file not found: {file_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Example endpoints for testing
@app.post("/predict/example", response_model=PredictionResponseChurn)
async def predict_example():
    """
    Example prediction for testing
    """
    example_customer = FeatureVectorChurn(
        monthly_fee=29.99,
        usage_hours=45.5,
        support_requests=2,
        account_age_months=12,
        failed_payments=0,
        region="europe",
        device_type="mobile",
        payment_method="card",
        autopay_enabled=1
    )

    return await predict_churn(example_customer, customer_id="example_customer")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )