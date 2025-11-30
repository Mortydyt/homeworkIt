"""
Application configuration
"""

import os
from typing import Optional


class Settings:
    """Application settings"""

    # App settings
    app_name: str = "Churn Prediction API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Model settings
    model_path: str = os.getenv("MODEL_PATH", "models/churn_model.pkl")
    default_dataset_path: str = os.getenv("DATASET_PATH", "churn_dataset.csv")

    # API settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    reload: bool = os.getenv("RELOAD", "false").lower() == "true"

    # ML settings
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "1000"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "info")

    @classmethod
    def get_settings(cls) -> "Settings":
        """Get settings instance"""
        return cls()


# Global settings instance
settings = Settings.get_settings()