"""
Churn prediction ML pipeline
Days 3-6: Data processing, model training, and prediction
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelPipeline:
    """Complete churn prediction pipeline"""

    def __init__(self, model_path: str = "models/churn_model.pkl"):
        """
        Initialize the churn model pipeline

        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Model components
        self.model = None
        self.preprocessor = None
        self.feature_columns = []
        self.categorical_columns = ['region', 'device_type', 'payment_method']
        self.numerical_columns = ['monthly_fee', 'usage_hours', 'support_requests',
                                'account_age_months', 'failed_payments', 'autopay_enabled']

        # Training metadata
        self.model_version = None
        self.training_time = None
        self.training_samples = 0
        self.test_samples = 0
        self.metrics = {}

        logger.info(f"ChurnModelPipeline initialized with model_path: {model_path}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load churn dataset from CSV file

        Args:
            file_path: Path to CSV file

        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

            # Validate required columns
            required_columns = self.numerical_columns + self.categorical_columns + ['churn']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        try:
            df_processed = df.copy()

            # Handle missing values
            numeric_impute_values = {
                'monthly_fee': df_processed['monthly_fee'].median(),
                'usage_hours': df_processed['usage_hours'].median(),
                'support_requests': df_processed['support_requests'].median(),
                'account_age_months': df_processed['account_age_months'].median(),
                'failed_payments': 0,  # Most common value
                'autopay_enabled': 0    # Most common value
            }

            for col, value in numeric_impute_values.items():
                if col in df_processed.columns:
                    df_processed[col].fillna(value, inplace=True)

            # Categorical imputation
            for col in self.categorical_columns:
                if col in df_processed.columns:
                    df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'unknown', inplace=True)

            # Validate data ranges
            df_processed['monthly_fee'] = df_processed['monthly_fee'].clip(lower=0)
            df_processed['usage_hours'] = df_processed['usage_hours'].clip(lower=0)
            df_processed['support_requests'] = df_processed['support_requests'].clip(lower=0)
            df_processed['account_age_months'] = df_processed['account_age_months'].clip(lower=0)
            df_processed['failed_payments'] = df_processed['failed_payments'].clip(lower=0)
            df_processed['autopay_enabled'] = df_processed['autopay_enabled'].clip(0, 1)
            df_processed['churn'] = df_processed['churn'].clip(0, 1)

            # Ensure proper data types
            for col in self.categorical_columns:
                df_processed[col] = df_processed[col].astype(str)

            logger.info("Data preprocessing completed successfully")
            return df_processed

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline

        Returns:
            ColumnTransformer for preprocessing
        """
        try:
            # Numeric preprocessing: StandardScaler
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Categorical preprocessing: OneHotEncoder
            categorical_transformer = Pipeline(steps=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numerical_columns),
                    ('cat', categorical_transformer, self.categorical_columns)
                ]
            )

            logger.info("Preprocessor created successfully")
            return preprocessor

        except Exception as e:
            logger.error(f"Error creating preprocessor: {str(e)}")
            raise

    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the churn prediction model

        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility

        Returns:
            Training metrics
        """
        try:
            start_time = time.time()

            # Prepare features and target
            X = df[self.numerical_columns + self.categorical_columns]
            y = df['churn']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            self.training_samples = len(X_train)
            self.test_samples = len(X_test)

            logger.info(f"Training model with {self.training_samples} samples, testing with {self.test_samples}")

            # Create preprocessor
            self.preprocessor = self.create_preprocessor()

            # Create model pipeline
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state,
                    class_weight='balanced'
                ))
            ])

            # Train model
            self.model.fit(X_train, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }

            # Calculate feature importance
            feature_importance = self._get_feature_importance()

            # Store training metadata
            self.training_time = time.time() - start_time
            self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Save model
            self.save_model()

            training_result = {
                'model_version': self.model_version,
                'training_samples': self.training_samples,
                'test_samples': self.test_samples,
                'training_time_seconds': self.training_time,
                'metrics': self.metrics,
                'feature_importance': feature_importance
            }

            logger.info(f"Model training completed successfully. Accuracy: {self.metrics['accuracy']:.3f}")
            return training_result

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model

        Returns:
            Dictionary of feature importance scores
        """
        try:
            if not self.model or not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                return {}

            # Get feature names after preprocessing
            feature_names = (self.numerical_columns +
                           list(self.model.named_steps['preprocessor']
                                .named_transformers_['cat']
                                .named_steps['encoder']
                                .get_feature_names_out(self.categorical_columns)))

            # Get feature importance
            importances = self.model.named_steps['classifier'].feature_importances_

            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importances))

            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(),
                                           key=lambda x: x[1], reverse=True))

            return feature_importance

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}

    def predict(self, customer_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for new customers

        Args:
            customer_data: List of customer feature dictionaries

        Returns:
            List of prediction results
        """
        try:
            if not self.model:
                raise ValueError("Model not trained. Call train_model() first.")

            # Convert to DataFrame
            df = pd.DataFrame(customer_data)

            # Validate required columns
            required_columns = self.numerical_columns + self.categorical_columns
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Make predictions
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)

            # Create prediction results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                churn_prob = prob[1]  # Probability of churn (class 1)
                risk_level = self._get_risk_level(churn_prob)
                recommendations = self._get_recommendations(churn_prob, customer_data[i])

                result = {
                    'churn_prediction': int(pred),
                    'churn_probability': float(churn_prob),
                    'confidence_score': float(max(prob)),
                    'risk_level': risk_level,
                    'recommendations': recommendations
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def _get_risk_level(self, churn_probability: float) -> str:
        """
        Determine risk level based on churn probability

        Args:
            churn_probability: Probability of churn

        Returns:
            Risk level string
        """
        if churn_probability < 0.3:
            return "low"
        elif churn_probability < 0.7:
            return "medium"
        else:
            return "high"

    def _get_recommendations(self, churn_probability: float, customer_data: Dict[str, Any]) -> List[str]:
        """
        Get recommendations based on churn probability and customer data

        Args:
            churn_probability: Probability of churn
            customer_data: Customer feature data

        Returns:
            List of recommendations
        """
        recommendations = []

        if churn_probability < 0.3:
            recommendations.append("Customer has low churn risk - maintain current service")
        elif churn_probability < 0.7:
            recommendations.append("Monitor customer usage patterns")
            if customer_data.get('support_requests', 0) > 2:
                recommendations.append("Proactive support outreach recommended")
            if customer_data.get('autopay_enabled', 0) == 0:
                recommendations.append("Offer autopay setup discount")
        else:
            recommendations.append("High churn risk - immediate intervention needed")
            if customer_data.get('support_requests', 0) > 3:
                recommendations.append("Assign dedicated support representative")
            if customer_data.get('failed_payments', 0) > 1:
                recommendations.append("Review payment issues and offer flexible payment options")
            recommendations.append("Consider retention offers or discounts")

        return recommendations

    def save_model(self) -> None:
        """Save the trained model and metadata"""
        try:
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'feature_columns': self.feature_columns,
                'numerical_columns': self.numerical_columns,
                'categorical_columns': self.categorical_columns,
                'model_version': self.model_version,
                'training_time': self.training_time,
                'training_samples': self.training_samples,
                'test_samples': self.test_samples,
                'metrics': self.metrics
            }

            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self) -> bool:
        """
        Load a trained model

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False

            model_data = joblib.load(self.model_path)

            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_columns = model_data.get('feature_columns', [])
            self.model_version = model_data.get('model_version')
            self.training_time = model_data.get('training_time')
            self.training_samples = model_data.get('training_samples', 0)
            self.test_samples = model_data.get('test_samples', 0)
            self.metrics = model_data.get('metrics', {})

            logger.info(f"Model loaded from {self.model_path}, version: {self.model_version}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status

        Returns:
            Model status dictionary
        """
        return {
            'model_trained': self.model is not None,
            'model_version': self.model_version,
            'last_training_time': datetime.fromtimestamp(self.training_time).isoformat() if self.training_time else None,
            'training_samples': self.training_samples,
            'model_metrics': self.metrics,
            'feature_columns': self.numerical_columns + self.categorical_columns
        }