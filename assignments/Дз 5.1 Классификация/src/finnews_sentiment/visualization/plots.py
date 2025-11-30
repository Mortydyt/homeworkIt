"""
Visualization utilities for financial news sentiment analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_sentiment_distribution(df: pd.DataFrame, sentiment_column: str = 'sentiment',
                               save_path: Optional[str] = None) -> None:
    """
    Plot sentiment distribution

    Args:
        df: DataFrame with sentiment data
        sentiment_column: Name of sentiment column
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 8))

    # Create subplots
    plt.subplot(2, 2, 1)
    sentiment_counts = df[sentiment_column].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    # Add value labels on bars
    for bar, count in zip(bars, sentiment_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')

    # Pie chart
    plt.subplot(2, 2, 2)
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Sentiment Proportions')

    # Distribution by word count (if available)
    if 'word_count' in df.columns:
        plt.subplot(2, 2, 3)
        for sentiment in df[sentiment_column].unique():
            subset = df[df[sentiment_column] == sentiment]
            plt.hist(subset['word_count'], alpha=0.6, label=sentiment, bins=20)
        plt.title('Word Count Distribution by Sentiment')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.legend()

    # Text length distribution (if available)
    if 'char_count' in df.columns:
        plt.subplot(2, 2, 4)
        for sentiment in df[sentiment_column].unique():
            subset = df[df[sentiment_column] == sentiment]
            plt.hist(subset['char_count'], alpha=0.6, label=sentiment, bins=20)
        plt.title('Character Count Distribution by Sentiment')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sentiment distribution plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(y_true: List, y_pred: List, labels: List[str],
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save plot
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.5)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add metrics
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}',
             ha='center', transform=plt.gca().transAxes)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")

    plt.show()


def plot_feature_importance(importance_dict: Dict[str, float], top_n: int = 20,
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance

    Args:
        importance_dict: Dictionary of feature importance
        top_n: Number of top features to show
        save_path: Path to save plot
    """
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importance = zip(*sorted_features)

    plt.figure(figsize=(12, 8))

    # Create horizontal bar plot
    bars = plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {len(features)} Feature Importance')
    plt.gca().invert_yaxis()  # Display most important at top

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, importance)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training history

    Args:
        history: Dictionary with training metrics
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))

    # Plot metrics
    for metric, values in history.items():
        plt.plot(values, label=metric, marker='o')

    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")

    plt.show()


def plot_sentiment_over_time(df: pd.DataFrame, date_column: str, sentiment_column: str,
                             save_path: Optional[str] = None) -> None:
    """
    Plot sentiment trends over time

    Args:
        df: DataFrame with date and sentiment columns
        date_column: Name of date column
        sentiment_column: Name of sentiment column
        save_path: Path to save plot
    """
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Group by date and sentiment
    sentiment_over_time = df.groupby([date_column, sentiment_column]).size().unstack(fill_value=0)

    plt.figure(figsize=(15, 8))

    # Plot stacked area chart
    sentiment_over_time.plot(kind='area', stacked=True, alpha=0.7, figsize=(15, 8))
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.legend(title='Sentiment')
    plt.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sentiment over time plot saved to {save_path}")

    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create correlation heatmap of numerical features

    Args:
        df: DataFrame with numerical features
        save_path: Path to save plot
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if len(numerical_cols) == 0:
        logger.warning("No numerical columns found for correlation heatmap")
        return

    plt.figure(figsize=(12, 10))

    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()

    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": .8})

    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {save_path}")

    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: Optional[str] = None) -> None:
    """
    Compare multiple models

    Args:
        results: Dictionary with model results
        save_path: Path to save plot
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Extract scores
    model_scores = {metric: [] for metric in metrics}
    for model in models:
        model_results = results[model]
        for metric in metrics:
            model_scores[metric].append(model_results.get(metric, 0))

    plt.figure(figsize=(12, 8))

    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.2

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, model_scores[metric], width, label=metric.title(), color=colors[i])

    plt.title('Model Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(x + width*1.5, models)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, metric in enumerate(metrics):
        for j, value in enumerate(model_scores[metric]):
            plt.text(j + i*width, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing visualization functions...")

    # Create sample data
    sample_df = pd.DataFrame({
        'sentiment': ['positive', 'negative', 'neutral'] * 100,
        'word_count': np.random.randint(10, 100, 300),
        'char_count': np.random.randint(50, 500, 300)
    })

    # Test sentiment distribution plot
    print("Creating sentiment distribution plot...")
    plot_sentiment_distribution(sample_df)

    # Test correlation heatmap
    print("Creating correlation heatmap...")
    create_correlation_heatmap(sample_df)

    print("Visualization tests completed!")