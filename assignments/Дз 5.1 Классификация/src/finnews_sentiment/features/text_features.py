"""
Text feature extraction for financial news sentiment analysis
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Advanced text preprocessing for financial news"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Financial-specific stop words
        financial_stop_words = {
            'said', 'say', 'says', 'year', 'month', 'day', 'new', 'also',
            'inc', 'corp', 'ltd', 'co', 'company', 'companies', 'report',
            'reports', 'according', 'would', 'could', 'should', 'may'
        }
        self.stop_words.update(financial_stop_words)

        # Financial keywords for sentiment
        self.positive_keywords = {
            'growth', 'gain', 'rise', 'increase', 'profit', 'earnings', 'revenue',
            'bull', 'bullish', 'rally', 'surge', 'boom', 'expansion', 'strong',
            'upgrade', 'outperform', 'beat', 'exceed', 'record', 'high', 'success'
        }

        self.negative_keywords = {
            'loss', 'fall', 'decline', 'decrease', 'drop', 'crash', 'recession',
            'bear', 'bearish', 'slump', 'downturn', 'contraction', 'weak', 'downgrade',
            'underperform', 'miss', 'cut', 'layoff', 'bankruptcy', 'debt', 'crisis'
        }

        self.neutral_keywords = {
            'stable', 'steady', 'flat', 'unchanged', 'maintain', 'hold', 'neutral',
            'mixed', 'average', 'normal', 'typical', 'standard', 'expect', 'forecast'
        }

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text and return list of cleaned tokens

        Args:
            text: Input text

        Returns:
            List of cleaned tokens
        """
        if not text or not isinstance(text, str):
            return []

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove punctuation and numbers, keep only alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and len(token) > 2]

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def extract_sentiment_keywords(self, text: str) -> Dict[str, int]:
        """
        Count sentiment-related keywords in text

        Args:
            text: Input text

        Returns:
            Dictionary with keyword counts
        """
        tokens = self.preprocess_text(text)

        positive_count = sum(1 for token in tokens if token in self.positive_keywords)
        negative_count = sum(1 for token in tokens if token in self.negative_keywords)
        neutral_count = sum(1 for token in tokens if token in self.neutral_keywords)

        return {
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'neutral_keywords': neutral_count
        }

    def extract_financial_entities(self, text: str) -> Dict[str, int]:
        """
        Extract financial entities and terms

        Args:
            text: Input text

        Returns:
            Dictionary with entity counts
        """
        text_lower = text.lower()

        # Financial entities patterns
        patterns = {
            'companies': len(re.findall(r'\b(AAPL|GOOGL|MSFT|AMZN|TSLA|META|NVDA|JPM|BAC|WMT)\b', text_lower)),
            'currencies': len(re.findall(r'\b(USD|EUR|GBP|JPY|CNY|dollar|euro|pound|yen)\b', text_lower)),
            'percentages': len(re.findall(r'\b\d+\.?\d*%\b', text)),
            'money': len(re.findall(r'\$\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|k|m|b|t)\b', text_lower)),
            'dates': len(re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', text_lower))
        }

        return patterns


def extract_text_features(df: pd.DataFrame, text_column: str = 'text_clean') -> pd.DataFrame:
    """
    Extract comprehensive text features

    Args:
        df: Input DataFrame
        text_column: Name of text column

    Returns:
        DataFrame with extracted features
    """
    preprocessor = TextPreprocessor()

    features = df.copy()

    logger.info("Extracting basic text features...")
    # Basic text features
    features['char_count'] = features[text_column].str.len()
    features['word_count'] = features[text_column].str.split().str.len()
    features['sentence_count'] = features[text_column].apply(lambda x: len(sent_tokenize(str(x))))
    features['avg_word_length'] = features.apply(lambda row:
        np.mean([len(word) for word in str(row[text_column]).split()]) if str(row[text_column]).split() else 0, axis=1)
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    features['exclamation_count'] = features[text_column].str.count('!')
    features['question_count'] = features[text_column].str.count('?')
    features['uppercase_count'] = features[text_column].str.count(r'[A-Z]')

    logger.info("Extracting sentiment keywords...")
    # Sentiment keywords
    sentiment_features = features[text_column].apply(preprocessor.extract_sentiment_keywords)
    sentiment_df = pd.DataFrame(sentiment_features.tolist(), index=features.index)
    features = pd.concat([features, sentiment_df], axis=1)

    logger.info("Extracting financial entities...")
    # Financial entities
    financial_features = features[text_column].apply(preprocessor.extract_financial_entities)
    financial_df = pd.DataFrame(financial_features.tolist(), index=features.index)
    features = pd.concat([features, financial_df], axis=1)

    logger.info("Calculating TextBlob sentiment scores...")
    # TextBlob sentiment scores
    features['textblob_polarity'] = features[text_column].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if x else 0
    )
    features['textblob_subjectivity'] = features[text_column].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity if x else 0
    )

    # Readability score (simplified)
    features['readability_score'] = features.apply(lambda row:
        206.835 - 1.015 * (row['word_count'] / max(1, row['sentence_count'])) - 84.6 * (row['avg_word_length'] if row['avg_word_length'] > 0 else 0)
        , axis=1)

    logger.info("Extracting n-gram and TF-IDF features...")
    # TF-IDF features
    try:
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )

        # Fill NaN values
        text_data = features[text_column].fillna('')
        tfidf_matrix = vectorizer.fit_transform(text_data)

        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{term}' for term in vectorizer.get_feature_names_out()],
            index=features.index
        )

        features = pd.concat([features, tfidf_df], axis=1)

        logger.info(f"Extracted {len(vectorizer.get_feature_names_out())} TF-IDF features")

    except Exception as e:
        logger.warning(f"TF-IDF extraction failed: {str(e)}")

    logger.info(f"Text feature extraction completed. Total features: {len(features.columns)}")

    return features


def create_vocabulary(df: pd.DataFrame, text_column: str = 'text_clean',
                     min_freq: int = 5, max_features: int = 10000) -> Dict[str, int]:
    """
    Create vocabulary from training data

    Args:
        df: Training DataFrame
        text_column: Text column name
        min_freq: Minimum frequency for words
        max_features: Maximum vocabulary size

    Returns:
        Dictionary mapping words to indices
    """
    preprocessor = TextPreprocessor()

    # Process all texts
    all_tokens = []
    for text in df[text_column].fillna(''):
        tokens = preprocessor.preprocess_text(text)
        all_tokens.extend(tokens)

    # Count word frequencies
    word_counts = Counter(all_tokens)

    # Filter by frequency and size
    vocabulary = {
        word: idx + 2  # 0=PAD, 1=UNK
        for idx, (word, count) in enumerate(word_counts.most_common(max_features))
        if count >= min_freq
    }

    logger.info(f"Created vocabulary with {len(vocabulary)} words")

    return vocabulary


def texts_to_sequences(texts: List[str], vocabulary: Dict[str, int],
                       max_length: int = 100) -> np.ndarray:
    """
    Convert texts to sequences of integers

    Args:
        texts: List of texts
        vocabulary: Word to index mapping
        max_length: Maximum sequence length

    Returns:
        Array of sequences
    """
    preprocessor = TextPreprocessor()

    sequences = []
    for text in texts:
        tokens = preprocessor.preprocess_text(text)
        sequence = [vocabulary.get(token, 1) for token in tokens[:max_length]]

        # Pad sequence
        if len(sequence) < max_length:
            sequence.extend([0] * (max_length - len(sequence)))

        sequences.append(sequence)

    return np.array(sequences)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Stock market reaches new all-time high as tech stocks surge!",
        "Economy faces serious challenges with rising inflation and unemployment.",
        "Market remains stable as Federal Reserve maintains current interest rates.",
        "Investors show mixed reactions to latest earnings reports.",
        "Banking sector experiences significant losses amid economic uncertainty."
    ]

    df = pd.DataFrame({'text_clean': sample_texts})

    print("Extracting text features...")
    features_df = extract_text_features(df)

    print(f"Original texts: {len(df)}")
    print(f"Features extracted: {len(features_df.columns)}")
    print("\nFeature types:")
    print(features_df.dtypes.value_counts())