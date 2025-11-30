"""
Sentiment classification models for financial news
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class SentimentClassifier:
    """Ensemble sentiment classifier for financial news"""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize sentiment classifier

        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = SVC(
                random_state=42,
                probability=True,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type} sentiment classifier")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the sentiment classifier

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training results and metrics
        """
        logger.info(f"Training {self.model_type} sentiment classifier...")

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_

        # Train model
        self.model.fit(X_train, y_train_encoded)

        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train_encoded, train_pred)

        results = {
            'model_type': self.model_type,
            'classes': self.classes_.tolist(),
            'train_accuracy': train_accuracy,
            'label_encoder': self.label_encoder
        }

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val_encoded, val_pred)

            results['val_accuracy'] = val_accuracy
            results['classification_report'] = classification_report(
                y_val_encoded, val_pred, target_names=self.classes_, output_dict=True
            )
            results['confusion_matrix'] = confusion_matrix(y_val_encoded, val_pred)

            logger.info(f"Training accuracy: {train_accuracy:.3f}")
            logger.info(f"Validation accuracy: {val_accuracy:.3f}")

        self.is_trained = True
        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict sentiment labels

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Features

        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (if available)

        Returns:
            Dictionary of feature importance
        """
        if not self.is_trained:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict

        return None

    def save_model(self, path: str) -> None:
        """
        Save trained model

        Args:
            path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'classes': self.classes_,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load trained model

        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.classes_ = model_data['classes']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {path}")


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame = None, y_val: pd.Series = None,
                model_types: List[str] = None) -> Dict[str, Any]:
    """
    Train sentiment classification model with hyperparameter tuning

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_types: List of model types to try

    Returns:
        Training results and best model
    """
    if model_types is None:
        model_types = ['random_forest', 'logistic_regression']

    best_model = None
    best_score = 0
    best_results = None

    logger.info(f"Training sentiment models: {model_types}")

    for model_type in model_types:
        try:
            logger.info(f"Training {model_type}...")

            # Initialize classifier
            classifier = SentimentClassifier(model_type)

            # Hyperparameter tuning
            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            else:
                # For other models, use default parameters
                param_grid = {}

            # Perform grid search if parameters defined
            if param_grid and X_val is not None:
                y_train_encoded = classifier.label_encoder.fit_transform(y_train)
                y_val_encoded = classifier.label_encoder.transform(y_val)

                grid_search = GridSearchCV(
                    classifier.model,
                    param_grid,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1
                )

                grid_search.fit(X_train, y_train_encoded)

                classifier.model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_

                # Evaluate on validation set
                val_pred = classifier.model.predict(X_val)
                val_score = accuracy_score(y_val_encoded, val_pred)

            else:
                # Simple training without hyperparameter tuning
                results = classifier.train(X_train, y_train, X_val, y_val)
                val_score = results.get('val_accuracy', 0)
                cv_score = results.get('train_accuracy', 0)
                best_params = {}

            # Track best model
            if val_score > best_score:
                best_score = val_score
                best_model = classifier
                best_results = {
                    'model_type': model_type,
                    'val_accuracy': val_score,
                    'cv_score': cv_score,
                    'best_params': best_params,
                    'model': classifier
                }

            logger.info(f"{model_type} - Validation accuracy: {val_score:.3f}")

        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            continue

    if best_model is None:
        raise ValueError("No model was successfully trained")

    logger.info(f"Best model: {best_results['model_type']} with accuracy: {best_score:.3f}")

    return best_results


def predict_sentiment(texts: List[str], model_path: str,
                      feature_extractor: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Predict sentiment for new texts

    Args:
        texts: List of texts to analyze
        model_path: Path to trained model
        feature_extractor: Feature extraction function

    Returns:
        List of prediction results
    """
    # Load model
    classifier = SentimentClassifier()
    classifier.load_model(model_path)

    # Extract features if extractor provided
    if feature_extractor:
        df = pd.DataFrame({'text': texts})
        features = feature_extractor(df)
        X = features.drop(['text'], axis=1, errors='ignore')
    else:
        # Assume texts are already feature vectors
        X = pd.DataFrame(texts)

    # Make predictions
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    # Prepare results
    results = []
    for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
        result = {
            'text': text,
            'predicted_sentiment': pred,
            'confidence': float(max(probs)),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(classifier.classes_, probs)
            }
        }
        results.append(result)

    return results


if __name__ == "__main__":
    # Example usage
    print("Testing sentiment classifier...")

    # Sample data
    sample_texts = [
        "Stock market reaches new all-time high",
        "Economy faces serious challenges ahead",
        "Market remains stable for now"
    ]

    sample_labels = ['positive', 'negative', 'neutral']

    # Create dummy features for testing
    X_dummy = pd.DataFrame(np.random.rand(len(sample_texts), 10))
    y_dummy = pd.Series(sample_labels)

    # Train model
    results = train_model(X_dummy, y_dummy)
    print(f"Training completed: {results['model_type']}")
    print(f"Validation accuracy: {results['val_accuracy']:.3f}")