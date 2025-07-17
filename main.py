import logging
import os
import sys
from preprocessing.preprocessing import DataPreprocessor
from feature_engineering.feat_eng import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("="*50 + "\n")

def main():
    try:
        # Preprocessing
        preprocessor = DataPreprocessor('amazon_usecase/data/amazon_reviews_us_Books_v1_02.tsv')
        df_clean = preprocessor.run_pipeline()
        print(df_clean.shape)

        # Feature Engineering
        engineer = FeatureEngineer(df_clean)
        df_features = engineer.run_pipeline()
        
        # Save processed data
        # df_features.to_parquet('../data/processed_reviews.parquet')
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()