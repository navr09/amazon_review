import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import logging
import os
import pickle
from typing import Tuple, Dict
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
# from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HelpfulnessPredictor:
    """Review helpfulness predictor with probability outputs"""
    
    def __init__(self, model_type='gbm', random_state=42):
        self.random_state = random_state
        self.model = None
        self.cv_scores = None
        self.model_type = model_type.lower()
        self.X_test = None
        self.y_test = None
        
        # Model configurations
        self.config = {
            'gbm': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': random_state
                }
            },
            'xgb': {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'eval_metric': 'logloss',
                    # 'early_stopping_rounds': 10,
                    'random_state': random_state
                }
            },
            # 'lgbm': {
            #     'class': LGBMClassifier,
            #     'params': {
            #         'n_estimators': 200,
            #         'max_depth': 5,
            #         'learning_rate': 0.1,
            #         'subsample': 0.8,
            #         'objective': 'binary',
            #         'random_state': random_state
            #     }
            # }
        }

    def _get_base_model(self):
        """Initialize the selected model type"""
        model_config = self.config[self.model_type]
        return model_config['class'](**model_config['params'])

    def _build_pipeline(self,X) -> Pipeline:
        """Build the complete ML pipeline"""
        numeric_features = [
            'helpfulness_ratio', 'title_length', 'headline_length', 'body_length'
            , 'body_word_count', 'exclamation_count', 'question_count', 'polarity', 'subjectivity'
            , 'helpful_kw_ratio', 'helpfulness_signal', 'enhanced_helpfulness']
        
        categorical_features = ['star_rating','verified_purchase']
        numeric_features = [f for f in X.columns.tolist() if f in numeric_features]
        categorical_features = [f for f in X.columns.tolist() if f in categorical_features]

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
        
        base_model = self._get_base_model()
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', CalibratedClassifierCV(base_model, cv=3, method='sigmoid'))
        ])

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """
        Train model with cross-validation and holdout test set
        Args:
            test_size: Fraction of data to reserve for testing (default: 0.2)
        Returns:
            dict: Training metrics including CV and test scores
        """
        try:
            # 1. Train-test split
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state, 
                stratify=y
            )

            # Writing the files for faster use
            X_train.to_csv('amazon_usecase/data/X_train.csv',index=False)
            self.X_test.to_csv('amazon_usecase/data/X_test.csv',index=False)
            y_train.to_csv('amazon_usecase/data/y_train.csv',index=False)
            self.y_test.to_csv('amazon_usecase/data/y_test.csv',index=False)
            logger.info("Done writing train and test files")

            # 2. Initialize pipeline
            self.model = self._build_pipeline(X)
            
            # 3. Cross-validation on training set
            logger.info("Running 5-fold cross-validation...")
            self.cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            # 4. Train final model on full training set
            logger.info("Training final model on full training data...")
            self.model.fit(X_train, y_train)
            
            # 5. Evaluate on holdout test set
            test_proba = self.predict_proba(self.X_test)
            test_auc = roc_auc_score(self.y_test, test_proba)
            
            logger.info(f"cv_mean_auc: {np.mean(self.cv_scores)} \
                        ,cv_std_auc: {np.std(self.cv_scores)} \
                        ,cv_scores: {self.cv_scores.tolist()}, \
                test_auc: {test_auc}, \
                test_size: {test_size}, \
                n_train_samples: {len(X_train)}, \
                n_test_samples: {len(self.X_test)}")
            
            return self.X_test, self.y_test
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability scores for the positive class"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]

    # def evaluate_test_set(self) -> Dict:
    #     """Evaluate performance on holdout test set"""
    #     if self.X_test is None or self.y_test is None:
    #         raise ValueError("No test set available. Did you call train() with test_size > 0?")
        
    #     test_proba = self.predict_proba(self.X_test)
    #     return {
    #         'test_auc': roc_auc_score(self.y_test, test_proba),
    #         'test_size': len(self.X_test)
    #     }

    def save_model(self, filepath):
        """Save entire object using pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model pickled to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load pickled model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def plot_shap_importance(self, X_test: pd.DataFrame, save_path: str = None):
        """
        Generate SHAP importance plots for the trained model
        
        Args:
            X_test: Test data to calculate SHAP values on
            save_path: Optional path to save the plot
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet. Call train() first.")
                
            logger.info("Generating SHAP explanations...")
            
            # 1. Get the trained XGBoost model from the calibrated classifier
            calibrated_classifier = self.model.named_steps['classifier']
            xgb_model = calibrated_classifier.calibrated_classifiers_[0].estimator
            
            # 2. Transform the test data
            preprocessor = self.model.named_steps['preprocessor']
            X_test_transformed = preprocessor.transform(X_test)
            
            # 3. Get feature names after preprocessing
            num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names = np.concatenate([num_features, cat_features])
            
            # 4. Create SHAP explainer
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test_transformed)
            
            # 5. Generate and save plots
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_transformed, 
                            feature_names=feature_names,
                            show=False)
            plt.title("SHAP Feature Importance")
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"SHAP plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")
            raise