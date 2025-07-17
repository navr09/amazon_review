import pandas as pd
from datetime import datetime
from typing import List, Tuple
import logging
from utils.column_converter import convert_columns
from utils.text_cleaner import clean_review_text
import numpy as np

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data with error handling"""
        try:
            self.df = pd.read_csv(
                self.filepath,
                sep='\t',
                on_bad_lines='skip',
                # dtype_backend='pyarrow'  # Faster parsing
            )
            logger.info(f"Successfully loaded data with {len(self.df)} rows")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def convert_columns(self, conversion_rules: List[Tuple[List[str], str]]) -> pd.DataFrame:
        """Apply column type conversions"""
        try:
            self.df = convert_columns(self.df, conversion_rules)
            logger.info("Column conversions completed")
            return self.df
        except Exception as e:
            logger.error(f"Column conversion failed: {str(e)}")
            raise

    def filter_by_date(self, start_year: int = 2003, end_year: int = 2005) -> pd.DataFrame:
        """Filter data by date range"""
        try:
            self.df = self.df[(self.df['review_date'].dt.year >= start_year) & (self.df['review_date'].dt.year <= end_year)]
            # Adding filter for total votes >3
            self.df = self.df[self.df['total_votes']>=3]
            logger.info(f"Filtered to {len(self.df)} rows between {start_year}-{end_year}")
            return self.df
        except Exception as e:
            logger.error(f"Date filtering failed: {str(e)}")
            raise

    def target_creation(self):
        try:
            self.df['helpfulness_ratio'] = np.where(
            self.df['total_votes'] > 0,
            self.df['helpful_votes'] / self.df['total_votes'],
            np.nan)
            logger.info(f"Created target column: helpfulness_ratio")
            return self.df
        except Exception as e:
            logger.error(f"Date filtering failed: {str(e)}")
            raise
    
    def clean_review(self):
        try:
            self.df['cleaned_review'] = self.df['review_body'].apply(clean_review_text)
            logger.info(f"Processed review body")
            return self.df
        except Exception as e:
            logger.error(f"Date filtering failed: {str(e)}")
            raise

    def remove_duplicates(self, 
                         text_dupe_threshold: int = 3,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate reviews
        Args:
            text_dupe_threshold: Remove reviews appearing more than this many times
            keep: Which duplicate to keep ('first', 'last', False)
        """
        try:
            high_freq_dupes = self.df['review_body'].value_counts()[self.df['review_body'].value_counts() > text_dupe_threshold].index
            df_clean = self.df[~self.df['review_body'].isin(high_freq_dupes)]
            # Keep only the first review when same user posts identical text
            df_sorted = df_clean.sort_values(by='helpfulness_ratio', ascending=False)
            # Step 2: Keep first occurrence of each duplicate review text
            df_clean = df_sorted.drop_duplicates(subset='review_body', keep=keep)
            # Step 3: Sort back to original order (optional)
            self.df = df_clean.sort_index()
            logger.info(f"Removed duplicates, remaining rows: {len(self.df)}")
            return self.df
        except Exception as e:
            logger.error(f"Duplicate removal failed: {str(e)}")
            raise

    def run_pipeline(self) -> pd.DataFrame:
        """Execute full preprocessing pipeline"""
        try:
            self.load_data()
            self.convert_columns([(['review_date'], 'datetime')])
            self.filter_by_date(2003, 2005)
            self.target_creation()
            self.clean_review()
            self.remove_duplicates()
            return self.df
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise