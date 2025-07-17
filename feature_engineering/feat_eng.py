import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
import logging
from utils.text_cleaner import clean_review_text

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.tfidf = None
        self.word_comparison = None

    def create_basic_features(self) -> pd.DataFrame:
        """Create basic text-based features"""
        try:
            # Length features
            self.df['title_length'] = self.df['product_title'].str.len()
            self.df['headline_length'] = self.df['review_headline'].str.len()
            self.df['body_length'] = self.df['review_body'].str.len()
            self.df['body_word_count'] = self.df['review_body'].str.split().str.len()
            
            # Sentiment analysis
            self.df['polarity'] = self.df['review_body'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            self.df['subjectivity'] = self.df['review_body'].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity
            )
            
            # Cleaned text
            self.df['cleaned_review'] = self.df['review_body'].apply(clean_review_text)
            
            logger.info("Basic text features created")
            return self.df
        except Exception as e:
            logger.error(f"Basic feature creation failed: {str(e)}")
            raise

    def create_tfidf_features(self, 
                            helpful_thresh: float = 0.7,
                            unhelpful_thresh: float = 0.3) -> pd.DataFrame:
        """Create TF-IDF based features"""
        try:
            # Initialize vectorizer
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                min_df=50,
                max_df=0.7,
                ngram_range=(1, 2),
                dtype=np.float32  # Reduce memory usage
            )
            
            # Fit on cleaned reviews
            self.tfidf.fit(self.df['cleaned_review'].astype(str))
            
            # Calculate helpfulness signals
            helpful = self.df[self.df['helpfulness_ratio'] > helpful_thresh]['cleaned_review']
            unhelpful = self.df[self.df['helpfulness_ratio'] < unhelpful_thresh]['cleaned_review']
            
            helpful_matrix = self.tfidf.transform(helpful)
            unhelpful_matrix = self.tfidf.transform(unhelpful)
            
            helpful_scores = np.asarray(helpful_matrix.mean(axis=0)).flatten()
            unhelpful_scores = np.asarray(unhelpful_matrix.mean(axis=0)).flatten()
            
            # Create word comparison
            self.word_comparison = pd.DataFrame({
                'word': self.tfidf.get_feature_names_out(),
                'helpful_score': helpful_scores,
                'unhelpful_score': unhelpful_scores
            }).set_index('word')
            
            # Filter artifacts
            self.word_comparison = self.word_comparison[
                ~self.word_comparison.index.str.contains(r'^[^a-zA-Z]|^[a-z]{1,2}$')
            ]
            
            # Add discrimination metric
            self.word_comparison['helpfulness_discrimination'] = (
                self.word_comparison['helpful_score'] - self.word_comparison['unhelpful_score']
            )
            
            logger.info("TF-IDF features processed")
            return self.df
        except Exception as e:
            logger.error(f"TF-IDF feature creation failed: {str(e)}")
            raise

    def create_keyword_features(self) -> pd.DataFrame:
        """Create keyword-based features"""
        try:
            if self.word_comparison is None:
                raise ValueError("Must run create_tfidf_features() first")
                
            # Get top keywords
            helpful_kws = self.word_comparison.nlargest(50, 'helpful_score').index.tolist()
            unhelpful_kws = self.word_comparison.nlargest(50, 'unhelpful_score').index.tolist()
            
            # Keyword presence
            self.df['contains_helpful_kw'] = self.df['cleaned_review'].str.contains(
                '|'.join(helpful_kws), case=False
            ).astype(int)
            
            self.df['contains_unhelpful_kw'] = self.df['cleaned_review'].str.contains(
                '|'.join(unhelpful_kws), case=False
            ).astype(int)
            
            # Keyword ratios
            self.df['helpful_kw_ratio'] = (
                self.df['cleaned_review'].str.count('|'.join(helpful_kws), flags=re.IGNORECASE) / 
                np.maximum(self.df['body_word_count'], 1)
            )
            
            logger.info("Keyword features created")
            return self.df
        except Exception as e:
            logger.error(f"Keyword feature creation failed: {str(e)}")
            raise

    def create_helpfulness_signal(self) -> pd.DataFrame:
        """Create final helpfulness signal feature"""
        try:
            if self.tfidf is None or self.word_comparison is None:
                raise ValueError("Must run create_tfidf_features() first")
                
            # Align weights with vocabulary
            vocab = set(self.tfidf.get_feature_names_out())
            word_comparison = self.word_comparison[self.word_comparison.index.isin(vocab)]
            word_weights = word_comparison['helpfulness_discrimination']
            word_weights = word_weights.reindex(self.tfidf.get_feature_names_out(), fill_value=0)
            
            # Transform and weight
            tfidf_features = self.tfidf.transform(self.df['cleaned_review'])
            weighted_tfidf = tfidf_features.multiply(word_weights.values)
            self.df['helpfulness_signal'] = np.array(weighted_tfidf.sum(axis=1)).flatten()
            
            logger.info("Helpfulness signal feature created")
            return self.df
        except Exception as e:
            logger.error(f"Helpfulness signal creation failed: {str(e)}")
            raise

    def run_pipeline(self) -> pd.DataFrame:
        """Execute full feature engineering pipeline"""
        try:
            self.create_basic_features()
            self.create_tfidf_features()
            self.create_keyword_features()
            self.create_helpfulness_signal()
            
            # Final cleanup
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df.fillna(0, inplace=True)
            
            logger.info(f"Feature engineering complete. Final shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {str(e)}")
            raise