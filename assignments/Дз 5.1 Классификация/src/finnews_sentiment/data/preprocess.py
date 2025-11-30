"""
Data preprocessing utilities for financial news sentiment analysis
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def clean_text_data(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Clean and preprocess text data

    Args:
        df: Input DataFrame
        text_column: Name of text column to clean

    Returns:
        DataFrame with cleaned text
    """
    df_clean = df.copy()

    # Find text column if not specified
    if text_column not in df_clean.columns:
        text_columns = ['text', 'content', 'article', 'headline', 'title', 'description']
        available_cols = [col for col in text_columns if col in df_clean.columns]
        if available_cols:
            text_column = available_cols[0]
            logger.info(f"Using column '{text_column}' as text column")
        else:
            raise ValueError("No text column found in DataFrame")

    # Apply text cleaning
    logger.info("Cleaning text data...")
    df_clean[f'{text_column}_clean'] = df_clean[text_column].progress_apply(clean_text)

    # Add text length features
    df_clean['original_length'] = df_clean[text_column].str.len()
    df_clean['clean_length'] = df_clean[f'{text_column}_clean'].str.len()
    df_clean['word_count'] = df_clean[f'{text_column}_clean'].str.split().str.len()

    # Filter out very short or very long texts
    min_length = 10
    max_length = 1000

    before_count = len(df_clean)
    df_clean = df_clean[
        (df_clean['clean_length'] >= min_length) &
        (df_clean['clean_length'] <= max_length)
    ]
    after_count = len(df_clean)

    logger.info(f"Filtered {before_count - after_count} texts due to length constraints")

    return df_clean


def clean_text(text: str) -> str:
    """
    Clean individual text string

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Remove numbers (keep financial numbers with decimal points)
    text = re.sub(r'\b\d+\.?\d*\b', '', text)

    # Remove punctuation except financial symbols
    punctuation_to_keep = '.%$€£¥'  # Keep decimal point and currency symbols
    punctuation_to_remove = ''.join([p for p in string.punctuation if p not in punctuation_to_keep])
    text = text.translate(str.maketrans('', '', punctuation_to_remove))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def validate_sentiment_labels(df: pd.DataFrame, label_column: str = 'sentiment',
                           valid_labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate and standardize sentiment labels

    Args:
        df: Input DataFrame
        label_column: Name of label column
        valid_labels: List of valid labels

    Returns:
        DataFrame with validated labels
    """
    if valid_labels is None:
        valid_labels = ['positive', 'negative', 'neutral']

    df_valid = df.copy()

    if label_column not in df_valid.columns:
        label_columns = ['sentiment', 'label', 'class', 'category']
        available_cols = [col for col in label_columns if col in df_valid.columns]
        if available_cols:
            label_column = available_cols[0]
            logger.info(f"Using column '{label_column}' as label column")
        else:
            raise ValueError("No label column found in DataFrame")

    # Standardize labels
    label_mapping = {
        'pos': 'positive', 'positive': 'positive', '1': 'positive', 1: 'positive',
        'neg': 'negative', 'negative': 'negative', '-1': 'negative', -1: 'negative', '0': 'negative', 0: 'negative',
        'neu': 'neutral', 'neutral': 'neutral', '2': 'neutral', 2: 'neutral'
    }

    before_count = len(df_valid)
    df_valid[label_column] = df_valid[label_column].map(lambda x: label_mapping.get(str(x).lower().strip(), str(x).lower().strip()))
    df_valid = df_valid[df_valid[label_column].isin(valid_labels)]
    after_count = len(df_valid)

    logger.info(f"Filtered {before_count - after_count} records due to invalid labels")
    logger.info(f"Label distribution:\n{df_valid[label_column].value_counts()}")

    return df_valid


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate records

    Args:
        df: Input DataFrame
        subset: Columns to check for duplicates

    Returns:
        DataFrame without duplicates
    """
    before_count = len(df)
    df_no_duplicates = df.drop_duplicates(subset=subset, keep='first')
    after_count = len(df_no_duplicates)

    logger.info(f"Removed {before_count - after_count} duplicate records")

    return df_no_duplicates


def balance_dataset(df: pd.DataFrame, label_column: str = 'sentiment',
                   method: str = 'undersample', random_state: int = 42) -> pd.DataFrame:
    """
    Balance the dataset by handling class imbalance

    Args:
        df: Input DataFrame
        label_column: Name of label column
        method: 'undersample', 'oversample', or 'hybrid'
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    from collections import Counter
    from sklearn.utils import resample

    # Count class distribution
    class_counts = Counter(df[label_column])
    logger.info(f"Original class distribution: {class_counts}")

    min_count = min(class_counts.values())
    max_count = max(class_counts.values())

    balanced_dfs = []

    for label in class_counts.keys():
        class_df = df[df[label_column] == label]

        if method == 'undersample':
            # Undersample majority classes
            if len(class_df) > min_count:
                class_df = resample(class_df,
                                   replace=False,
                                   n_samples=min_count,
                                   random_state=random_state)

        elif method == 'oversample':
            # Oversample minority classes
            if len(class_df) < max_count:
                class_df = resample(class_df,
                                   replace=True,
                                   n_samples=max_count,
                                   random_state=random_state)

        balanced_dfs.append(class_df)

    df_balanced = pd.concat(balanced_dfs)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    new_class_counts = Counter(df_balanced[label_column])
    logger.info(f"Balanced class distribution ({method}): {new_class_counts}")

    return df_balanced


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1,
               random_state: int = 42, stratify_column: str = 'sentiment') -> tuple:
    """
    Split data into train, validation, and test sets

    Args:
        df: Input DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        stratify_column: Column to stratify on

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_column] if stratify_column in df.columns else None
    )

    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_df[stratify_column] if stratify_column in train_val_df.columns else None
    )

    logger.info(f"Data split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def preprocess_pipeline(df: pd.DataFrame,
                       text_column: str = 'text',
                       label_column: str = 'sentiment',
                       remove_duplicates: bool = True,
                       balance_data: bool = True,
                       balance_method: str = 'undersample') -> tuple:
    """
    Complete preprocessing pipeline

    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        remove_duplicates: Whether to remove duplicates
        balance_data: Whether to balance the dataset
        balance_method: Method for balancing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Starting preprocessing pipeline...")

    # Step 1: Validate labels
    df = validate_sentiment_labels(df, label_column)

    # Step 2: Clean text data
    df = clean_text_data(df, text_column)

    # Step 3: Remove duplicates
    if remove_duplicates:
        df = remove_duplicates(df, subset=[text_column, label_column])

    # Step 4: Balance dataset
    if balance_data:
        df = balance_dataset(df, label_column, balance_method)

    # Step 5: Split data
    train_df, val_df, test_df = split_data(df, stratify_column=label_column)

    logger.info("Preprocessing pipeline completed successfully!")

    return train_df, val_df, test_df


# Add progress_apply to pandas for progress bars
tqdm.pandas()


if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'text': [
            "Stock market reaches new heights!",
            "Economy faces serious challenges.",
            "Market remains stable.",
            "Trading volume is average.",
            "Investors are concerned about inflation.",
            "Tech stocks perform well."
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'neutral', 'negative', 'positive']
    })

    print("Original data:")
    print(sample_data)

    # Apply preprocessing
    processed = preprocess_pipeline(sample_data)
    train_df, val_df, test_df = processed

    print(f"\nProcessed data:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")