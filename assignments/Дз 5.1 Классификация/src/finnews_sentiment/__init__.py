"""
Financial News Sentiment Analysis Package
"""

__version__ = "1.0.0"
__author__ = "Financial News Analysis Team"

from .data import load_data, preprocess_data
from .features import extract_features, TextPreprocessor
from .models import SentimentClassifier, train_model, predict
from .visualization import plot_sentiment_distribution, plot_confusion_matrix

__all__ = [
    "load_data",
    "preprocess_data",
    "extract_features",
    "TextPreprocessor",
    "SentimentClassifier",
    "train_model",
    "predict",
    "plot_sentiment_distribution",
    "plot_confusion_matrix"
]