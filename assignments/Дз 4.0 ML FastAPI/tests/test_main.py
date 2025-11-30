"""
Basic tests for the FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from ml.churn_model import ChurnModelPipeline

client = TestClient(app)


class TestBasicEndpoints:
    """Test basic API endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "ML churn service is running"}

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_status" in data
        assert "uptime_seconds" in data


class TestPredictionEndpoints:
    """Test prediction endpoints"""

    def test_predict_without_model(self):
        """Test prediction when model is not trained"""
        # Make sure model is not loaded
        if os.path.exists("models/churn_model.pkl"):
            os.remove("models/churn_model.pkl")

        customer_data = {
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

        response = client.post("/predict", json=customer_data)
        assert response.status_code == 503
        assert "Model not trained" in response.json()["detail"]

    def test_predict_invalid_data(self):
        """Test prediction with invalid data"""
        invalid_data = {
            "monthly_fee": -10,  # Invalid: negative fee
            "usage_hours": 45.5,
            "support_requests": 2,
            "account_age_months": 12,
            "failed_payments": 0,
            "region": "europe",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 1
        }

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_invalid_format(self):
        """Test batch prediction with invalid format"""
        invalid_request = {
            "customers": []  # Empty list - should fail validation
        }

        response = client.post("/predict/batch", json=invalid_request)
        assert response.status_code == 422


class TestModelEndpoints:
    """Test model management endpoints"""

    def test_model_status_untrained(self):
        """Test model status when model is not trained"""
        # Make sure model is not loaded
        if os.path.exists("models/churn_model.pkl"):
            os.remove("models/churn_model.pkl")

        response = client.get("/model/status")
        assert response.status_code == 200
        data = response.json()
        assert data["model_trained"] is False
        assert data["model_version"] is None

    def test_load_nonexistent_model(self):
        """Test loading non-existent model"""
        # Make sure model file doesn't exist
        if os.path.exists("models/churn_model.pkl"):
            os.remove("models/churn_model.pkl")

        response = client.post("/model/load")
        assert response.status_code == 404
        assert "No trained model found" in response.json()["detail"]


class TestDataEndpoints:
    """Test data management endpoints"""

    def test_data_upload_nonexistent_file(self):
        """Test uploading non-existent file"""
        response = client.post("/data/upload?file_path=nonexistent.csv")
        assert response.status_code == 404
        assert "Dataset file not found" in response.json()["detail"]

    def test_data_stats_nonexistent_file(self):
        """Test getting stats for non-existent file"""
        response = client.get("/data/stats?file_path=nonexistent.csv")
        assert response.status_code == 404
        assert "Dataset file not found" in response.json()["detail"]

    @pytest.mark.skipif(not os.path.exists("churn_dataset.csv"), reason="Dataset not available")
    def test_data_stats_valid_file(self):
        """Test getting stats for valid dataset file"""
        response = client.get("/data/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_rows" in data
        assert "columns" in data
        assert "churn_distribution" in data
        assert data["total_rows"] > 0


if __name__ == "__main__":
    pytest.main([__file__])