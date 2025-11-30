"""
Pydantic schemas for churn prediction
Days 2, 13: Feature vectors and response models
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class RegionEnum(str, Enum):
    """Customer region enumeration"""
    EUROPE = "europe"
    ASIA = "asia"
    AMERICA = "america"
    AFRICA = "africa"


class DeviceTypeEnum(str, Enum):
    """Device type enumeration"""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class PaymentMethodEnum(str, Enum):
    """Payment method enumeration"""
    CARD = "card"
    PAYPAL = "paypal"
    CRYPTO = "crypto"


class FeatureVectorChurn(BaseModel):
    """Input features for churn prediction (Day 2)"""
    monthly_fee: float = Field(..., gt=0, description="Monthly subscription fee")
    usage_hours: float = Field(..., ge=0, description="Hours of service usage in last month")
    support_requests: int = Field(..., ge=0, description="Number of support requests")
    account_age_months: int = Field(..., ge=0, description="Account age in months")
    failed_payments: int = Field(..., ge=0, description="Number of failed payments")
    region: RegionEnum = Field(..., description="Customer region")
    device_type: DeviceTypeEnum = Field(..., description="Primary device type")
    payment_method: PaymentMethodEnum = Field(..., description="Payment method")
    autopay_enabled: int = Field(..., ge=0, le=1, description="Autopay enabled (0 or 1)")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        schema_extra = {
            "example": {
                "monthly_fee": 29.99,
                "usage_hours": 45.5,
                "support_requests": 2,
                "account_age_months": 12,
                "failed_payments": 0,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 1
            }
        }


class DatasetRowChurn(FeatureVectorChurn):
    """Dataset row with target churn label (Day 2)"""
    churn: int = Field(..., ge=0, le=1, description="Churn label (1 if churned, 0 if stayed)")

    class Config:
        schema_extra = {
            "example": {
                "monthly_fee": 29.99,
                "usage_hours": 45.5,
                "support_requests": 2,
                "account_age_months": 12,
                "failed_payments": 0,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 1,
                "churn": 0
            }
        }


class PredictionResponseChurn(BaseModel):
    """Response model for churn prediction (Day 7)"""
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    churn_prediction: int = Field(..., ge=0, le=1, description="Predicted churn (1 if will churn, 0 if will stay)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations to reduce churn")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "cust_12345",
                "churn_prediction": 0,
                "churn_probability": 0.23,
                "confidence_score": 0.87,
                "risk_level": "low",
                "recommendations": ["Continue current service usage pattern"],
                "processing_time_ms": 15.4
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions (Day 11)"""
    customers: List[FeatureVectorChurn] = Field(..., min_items=1, max_items=1000, description="List of customer features")

    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "monthly_fee": 29.99,
                        "usage_hours": 45.5,
                        "support_requests": 2,
                        "account_age_months": 12,
                        "failed_payments": 0,
                        "region": "europe",
                        "device_type": "mobile",
                        "payment_method": "card",
                        "autopay_enabled": 1
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions (Day 11)"""
    predictions: List[PredictionResponseChurn] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total number of customers processed")
    churn_risk_summary: Dict[str, int] = Field(..., description="Summary of churn risks")
    average_churn_probability: float = Field(..., description="Average churn probability")
    processing_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [],
                "total_customers": 100,
                "churn_risk_summary": {"low": 60, "medium": 25, "high": 15},
                "average_churn_probability": 0.32,
                "processing_time_ms": 125.7
            }
        }


class TrainingResponse(BaseModel):
    """Response model for model training (Day 6)"""
    message: str = Field(..., description="Training status message")
    model_version: str = Field(..., description="Trained model version")
    training_samples: int = Field(..., description="Number of training samples")
    test_samples: int = Field(..., description="Number of test samples")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    precision: Optional[float] = Field(None, description="Model precision")
    recall: Optional[float] = Field(None, description="Model recall")
    f1_score: Optional[float] = Field(None, description="Model F1 score")
    auc_roc: Optional[float] = Field(None, description="Model AUC-ROC score")
    training_time_seconds: Optional[float] = Field(None, description="Training time in seconds")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")

    class Config:
        schema_extra = {
            "example": {
                "message": "Model trained successfully",
                "model_version": "v1.0.0",
                "training_samples": 800,
                "test_samples": 200,
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.82,
                "f1_score": 0.83,
                "auc_roc": 0.91,
                "training_time_seconds": 2.4,
                "feature_importance": {
                    "usage_hours": 0.35,
                    "support_requests": 0.28,
                    "failed_payments": 0.22,
                    "monthly_fee": 0.15
                }
            }
        }


class ModelStatus(BaseModel):
    """Model status response (Day 12)"""
    model_trained: bool = Field(..., description="Whether model is trained")
    model_version: Optional[str] = Field(None, description="Current model version")
    last_training_time: Optional[str] = Field(None, description="Last training timestamp")
    training_samples: Optional[int] = Field(None, description="Number of training samples")
    model_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    feature_columns: List[str] = Field(..., description="Feature columns used by model")

    class Config:
        schema_extra = {
            "example": {
                "model_trained": True,
                "model_version": "v1.0.0",
                "last_training_time": "2024-01-15T10:30:00",
                "training_samples": 800,
                "model_metrics": {
                    "accuracy": 0.87,
                    "precision": 0.85,
                    "recall": 0.82,
                    "f1_score": 0.83,
                    "auc_roc": 0.91
                },
                "feature_columns": [
                    "monthly_fee",
                    "usage_hours",
                    "support_requests",
                    "account_age_months",
                    "failed_payments",
                    "region",
                    "device_type",
                    "payment_method",
                    "autopay_enabled"
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response (Day 13)"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_status: ModelStatus = Field(..., description="Model status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    timestamp: str = Field(..., description="Current timestamp")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_status": {
                    "model_trained": True,
                    "model_version": "v1.0.0"
                },
                "uptime_seconds": 3600.5,
                "memory_usage_mb": 128.7,
                "timestamp": "2024-01-15T12:00:00"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response (Day 10)"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "details": {
                    "field": "monthly_fee",
                    "issue": "value must be greater than 0"
                },
                "timestamp": "2024-01-15T12:00:00"
            }
        }