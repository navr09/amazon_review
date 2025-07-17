import logging
import os
import sys
from preprocessing.preprocessing import DataPreprocessor
from feature_engineering.feat_eng import FeatureEngineer
from modeling.model import HelpfulnessPredictor
from evaluation.eval import ModelEvaluator
from feature_selection.feat_sel import CliquesFeatureSelector
from utils.feature_split import feature_type_split
import pandas as pd
import numpy as np
import joblib



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("="*50 + "\n")

def main():
    try:
        # Preprocessing
        preprocessor = DataPreprocessor('amazon_usecase/data/amazon_reviews_us_Books_v1_02.tsv')
        df_clean = preprocessor.run_pipeline()

        # Feature Engineering
        engineer = FeatureEngineer(df_clean)
        df_features = engineer.run_pipeline()


        # Save processed data as it's time consuming
        df_features.to_csv('amazon_usecase/data/amazon_reviews_added_ftrs_updated_v2.csv',index=False)
        
        # df_features = pd.DataFrame(pd.read_csv('amazon_usecase/data/amazon_reviews_added_ftrs_updated_v2.csv'))
        # print(df_features.columns.tolist())
       
        # State meta columns
        meta_columns = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title'
                        , 'product_category','helpful_votes', 'total_votes', 'vine'
                        ,'allcaps_count','review_headline', 'review_body', 'review_date', 'cleaned_review'
                        ,'contains_helpful_kw', 'contains_unhelpful_kw']
        
        # 3. Prepare Modeling Data
        total_features = [col for col in df_features.columns.tolist() if col not in meta_columns]
        numerical_features,categorical_features = feature_type_split(df_features,total_features)


        # Add Feature selection and manual selection for categorical features to select final list of features
        selector = CliquesFeatureSelector(df_features[numerical_features], 'helpfulness_ratio', corr_thr=0.3, corr_type='pearson')
        selected_num_features = selector.get_selected_features()
        selected_num_features=selected_num_features['Features'].tolist()
        selected_cat_features = ['verified_purchase']

        # train_features = [f for f in selected_num_features+selected_cat_features if f != 'helpfulness_ratio']
        # print('train features:',train_features)
        # Hardcoding from feature selection exercise
        train_features = ['star_rating','headline_length',
                          'polarity','body_length','body_word_count','exclamation_count'
                          , 'helpfulness_signal', 'verified_purchase']
        X = df_features[train_features]
        y = (df_features['helpfulness_ratio'] >= 0.7).astype(int) # Target column

        # 4. Model Training with CV
        logger.info("Training model...")
        predictor = HelpfulnessPredictor(model_type='xgb')
        # Save the model for further evaluation and results
        X_test, y_test = predictor.train(X, y, test_size=0.2)
        predictor.save_model('xgbmodel.pkl')

        # Generate SHAP plot
        predictor.plot_shap_importance(
            X_test=X_test,
            save_path='amazon_usecase/results/plots/shap_importance.png'
        )

        # 4. Evaluate on test set
        logger.info("Testing model...")
        predictor = HelpfulnessPredictor.load_model('xgbmodel.pkl')
        X_test = pd.DataFrame(pd.read_csv('amazon_usecase/data/X_test.csv'))
        y_test = np.array(pd.read_csv('amazon_usecase/data/y_test.csv'))
        y_proba = predictor.predict_proba(X_test)
            # 1. Core metrics evaluation    
        metrics = ModelEvaluator.evaluate(y_test, y_proba)
        ModelEvaluator.plot_metrics(y_test, y_proba, save_path='amazon_usecase/results/plots/evaluation_plots.png')
        ModelEvaluator.save_metrics(metrics, 'amazon_usecase/results/evaluation_metrics.json')
            # 2. Drift analysis
        ModelEvaluator.plot_drift_analysis(
            y_test, 
            y_proba,
            save_path='amazon_usecase/results/plots/drift_analysis.png'
        )
            # 3. Threshold analysis
        ModelEvaluator.plot_optimal_threshold(
            y_test, 
            y_proba,
            save_path='amazon_usecase/results/plots/threshold_optimization.png'
        )
        ModelEvaluator.save_threshold_analysis(
            y_test, 
            y_proba,
            filepath='amazon_usecase/results/threshold_analysis.csv',
            min_threshold=0.3,
            max_threshold=0.8,
            step=0.05
        )
        
        # 4. Segment analysis
        ModelEvaluator.plot_segment_analysis(
            X_test, 
            y_test, 
            y_proba,
            segment_col='star_rating',
            save_path='amazon_usecase/results/plots/segment_analysis.png'
        )
        
        # 6. Generate Predictions
        # y_proba = predictor.predict_proba(X)
        
        logging.info("Pipeline completed successfully")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()