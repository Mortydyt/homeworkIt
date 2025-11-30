"""
Data loading and saving utilities for financial news sentiment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_news_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load financial news data from CSV or other formats

    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas.read_csv()

    Returns:
        DataFrame with news data
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Determine file format and load accordingly
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path, **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Successfully loaded {len(df)} records from {file_path}")

        # Validate basic structure
        _validate_basic_structure(df)

        return df

    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def _validate_basic_structure(df: pd.DataFrame) -> None:
    """
    Validate basic structure of news dataframe

    Args:
        df: DataFrame to validate
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Check for common column names
    text_columns = ['text', 'content', 'article', 'headline', 'title', 'description']
    label_columns = ['sentiment', 'label', 'class', 'category']

    available_text_cols = [col for col in text_columns if col in df.columns]
    available_label_cols = [col for col in label_columns if col in df.columns]

    if not available_text_cols:
        logger.warning("No standard text column found. Available columns: " + str(df.columns.tolist()))

    if not available_label_cols:
        logger.warning("No standard label column found. Available columns: " + str(df.columns.tolist()))


def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save DataFrame to file

    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments for pandas saving functions
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() == '.csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df.to_json(file_path, **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            df.to_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Successfully saved {len(df)} records to {file_path}")

    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise


def create_sample_dataset(output_path: str, num_samples: int = 1000) -> pd.DataFrame:
    """
    Create a sample financial news dataset for testing

    Args:
        output_path: Path to save the sample dataset
        num_samples: Number of samples to generate

    Returns:
        Generated DataFrame
    """
    import random

    # Sample positive financial news phrases
    positive_news = [
        "Stock market reaches new all-time high",
        "Company reports record quarterly earnings",
        "Federal Reserve signals economic recovery",
        "Tech sector leads market rally",
        "Investor confidence grows as economy improves",
        "Banking stocks surge on positive outlook",
        "GDP growth exceeds analyst expectations",
        "Unemployment rate falls to historic low"
    ]

    # Sample negative financial news phrases
    negative_news = [
        "Market crashes on recession fears",
        "Company misses earnings expectations badly",
        "Inflation reaches decade-high levels",
        "Tech stocks plummet amid sell-off",
        "Federal Reserve raises interest rates sharply",
        "Banking crisis threatens global economy",
        "GDP contraction deepens economic worries",
        "Unemployment spikes as companies cut jobs"
    ]

    # Sample neutral financial news phrases
    neutral_news = [
        "Fed maintains current interest rates",
        "Company announces quarterly results tomorrow",
        "Market awaits key economic data release",
        "Trading volume remains average for session",
        "Analysts maintain neutral rating on stock",
        "Central bank officials give policy speech",
        "Company announces shareholder meeting date",
        "Market opens flat following mixed signals"
    ]

    # Generate sample data
    data = []
    sentiments = []
    dates = []

    base_date = datetime(2024, 1, 1)

    for i in range(num_samples):
        # Randomly select sentiment
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        sentiments.append(sentiment)

        # Select corresponding news text
        if sentiment == 'positive':
            text = random.choice(positive_news)
        elif sentiment == 'negative':
            text = random.choice(negative_news)
        else:
            text = random.choice(neutral_news)

        # Add some variation
        if random.random() > 0.5:
            text = f"{text}. Trading volume was {'high' if random.random() > 0.5 else 'low'}."

        data.append(text)

        # Generate random date within the last year
        days_offset = random.randint(0, 365)
        date = base_date + pd.Timedelta(days=days_offset)
        dates.append(date.strftime('%Y-%m-%d'))

    # Create DataFrame
    df = pd.DataFrame({
        'text': data,
        'sentiment': sentiments,
        'date': dates,
        'source': ['Financial News API'] * num_samples
    })

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Save the dataset
    save_data(df, output_path)

    logger.info(f"Created sample dataset with {num_samples} records at {output_path}")

    return df


def load_and_preprocess_data(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and perform basic preprocessing on news data

    Args:
        input_path: Path to raw data
        output_path: Optional path to save processed data

    Returns:
        Processed DataFrame
    """
    # Load data
    df = load_news_data(input_path)

    # Basic preprocessing
    from .preprocess import clean_text_data

    df_clean = clean_text_data(df)

    # Save processed data if path provided
    if output_path:
        save_data(df_clean, output_path)

    return df_clean


if __name__ == "__main__":
    # Create sample dataset for demonstration
    sample_path = "../../data/raw/sample_financial_news.csv"
    create_sample_dataset(sample_path, num_samples=500)