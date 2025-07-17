import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from typing import Tuple, Dict
from sklearn.pipeline import Pipeline 
import shap


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation on test set"""
    @staticmethod
    def _ensure_1d(y):
        """Convert input to 1D array/pd.Series"""
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if isinstance(y, np.ndarray):
            if y.ndim > 1:
                y = y.ravel()  # Flatten to 1D
        return pd.Series(y) if not isinstance(y, pd.Series) else y
    
    @staticmethod
    def generate_threshold_analysis(y_true, y_proba: np.ndarray, 
                                  min_threshold: float = 0.3, 
                                  max_threshold: float = 0.8, 
                                  step: float = 0.005) -> pd.DataFrame:
        """
        Generate CSV with classification metrics at different thresholds
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            min_threshold: Minimum threshold (default: 0.3)
            max_threshold: Maximum threshold (default: 0.8)
            step: Threshold increment step (default: 0.005)
            
        Returns:
            DataFrame with metrics at each threshold
        """
        y_true = ModelEvaluator._ensure_1d(y_true)
        results = []
        
        for threshold in np.arange(min_threshold, max_threshold + step, step):
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            results.append({
                'threshold': round(threshold, 3),
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn,
                'precision': tp / (tp + fp + 1e-8),
                'recall': tp / (tp + fn + 1e-8),
                'f1_score': 2 * tp / (2 * tp + fp + fn + 1e-8)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = None) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        Args:
            threshold: Optional decision threshold (default: optimal F1 threshold)
        Returns:
            dict: Complete evaluation metrics
        """
        # Calculate optimal threshold if not provided
        if threshold is None:
            threshold = ModelEvaluator.find_optimal_threshold(y_true, y_proba)
        
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_proba),
            'auc_pr': average_precision_score(y_true, y_proba),
            'threshold': threshold,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'precision_recall_curve': ModelEvaluator._get_curve_points(y_true, y_proba),
            'roc_curve': ModelEvaluator._get_roc_points(y_true, y_proba)
        }
        
        # Log key metrics
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"Optimal Threshold: {metrics['threshold']:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_true, y_pred))
        
        return metrics

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find threshold that maximizes F1 score"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        return thresholds[np.argmax(f1_scores)]

    @staticmethod
    def _get_curve_points(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Generate precision-recall curve data points"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        }

    @staticmethod
    def _get_roc_points(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Generate ROC curve data points"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }

    @staticmethod
    def plot_metrics(y_true: np.ndarray, y_proba: np.ndarray, save_path: str = None):
        """Generate and save evaluation plots"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Convert to hashable type if needed
        y_true = ModelEvaluator._ensure_1d(y_true)

        # ROC Curve
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax1)
        ax1.set_title('ROC Curve')
        
        # Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax2)
        ax2.set_title('Precision-Recall Curve')
        
        # Confusion Matrix
        threshold = ModelEvaluator.find_optimal_threshold(y_true, y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                   xticklabels=['Not Helpful', 'Helpful'],
                   yticklabels=['Not Helpful', 'Helpful'])
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved evaluation plots to {save_path}")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def save_metrics(metrics: Dict, filepath: str):
        """Save evaluation metrics to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")
    
    @staticmethod
    def save_threshold_analysis(y_true, y_proba: np.ndarray, 
                              filepath: str,
                              min_threshold: float = 0.3,
                              max_threshold: float = 0.8,
                              step: float = 0.005):
        """
        Generate and save threshold analysis to CSV
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            filepath: Path to save CSV
            min_threshold: Minimum threshold (default: 0.3)
            max_threshold: Maximum threshold (default: 0.8)
            step: Threshold increment step (default: 0.005)
        """
        df = ModelEvaluator.generate_threshold_analysis(
            y_true, y_proba, min_threshold, max_threshold, step
        )
        df.to_csv(filepath, index=False)
        logger.info(f"Threshold analysis saved to {filepath}")

    @staticmethod
    # def plot_drift_analysis(y_true_ratio, y_pred_proba, save_path=None):
    #     """
    #     Plot comparison between helpfulness ratio and predicted probabilities.
        
    #     Args:
    #         y_true_ratio: Actual helpfulness ratio (continuous 0-1)
    #         y_pred_proba: Model's predicted probabilities (0-1)
    #         save_path: Optional path to save the plot
    #     """
    #     plt.figure(figsize=(12, 5))
        
    #     # Distribution plot
    #     plt.subplot(1, 2, 1)
    #     sns.kdeplot(y_true_ratio, label='Actual Ratio', fill=True)
    #     sns.kdeplot(y_pred_proba, label='Predicted Probas', fill=True)
    #     plt.title("Distribution Comparison")
    #     plt.xlabel("Helpfulness Score")
    #     plt.legend()
        
    #     # Calibration curve
    #     plt.subplot(1, 2, 2)
    #     prob_true, prob_pred = calibration_curve(
    #         (y_true_ratio >= 0.7).astype(int),  # Binarized
    #         y_pred_proba,
    #         n_bins=10
    #     )
    #     plt.plot(prob_pred, prob_true, 's-', label='Model')
    #     plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    #     plt.xlabel("Predicted Probability")
    #     plt.ylabel("Actual Helpfulness Ratio")
    #     plt.title("Calibration Curve")
    #     plt.legend()
        
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #         plt.close()
    #     else:
    #         plt.show()
    def plot_drift_analysis(y_true_ratio, y_pred_proba, save_path=None):
        """
        Plot comparison between helpfulness ratio and predicted probabilities.
        
        Args:
            y_true_ratio: Actual helpfulness ratio (continuous 0-1)
            y_pred_proba: Model's predicted probabilities (0-1)
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(16, 5))
        
        # 1. Actual Helpfulness Ratio Bar Plot (New)
        plt.subplot(1, 3, 1)
        bins = np.linspace(0, 1, 11)  # 0-1 in 0.1 increments
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts, _ = np.histogram(y_true_ratio, bins=bins)
        
        # Normalize to percentage
        percentages = counts / len(y_true_ratio) * 100
        bars = plt.bar(bin_centers, percentages, width=0.08, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title("Actual Helpfulness Ratio Distribution")
        plt.xlabel("Helpfulness Ratio Bins")
        plt.ylabel("Percentage of Reviews")
        plt.xticks(bin_centers, [f"{x:.1f}" for x in bin_centers])
        plt.grid(axis='y', alpha=0.3)
        
        # 2. Distribution plot (KDE)
        plt.subplot(1, 3, 2)
        sns.kdeplot(y_true_ratio, label='Actual Ratio', fill=True)
        sns.kdeplot(y_pred_proba, label='Predicted Probas', fill=True)
        plt.title("Distribution Comparison")
        plt.xlabel("Helpfulness Score")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 3. Calibration curve
        plt.subplot(1, 3, 3)
        prob_true, prob_pred = calibration_curve(
            (y_true_ratio >= 0.7).astype(int),  # Binarized
            y_pred_proba,
            n_bins=10
        )
        plt.plot(prob_pred, prob_true, 's-', label='Model')
        plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Actual Helpfulness Ratio")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


    @staticmethod
    def plot_optimal_threshold(y_true, y_proba, save_path=None):
        """
        Plot precision, recall, and F1 across thresholds.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            save_path: Optional path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, precision[:-1], label='Precision')
        plt.plot(thresholds, recall[:-1], label='Recall')
        plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
        plt.axvline(thresholds[optimal_idx], color='r', linestyle='--', 
                   label=f'Optimal Threshold: {thresholds[optimal_idx]:.2f}')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Optimization")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_segment_analysis(df, y_true, y_proba, segment_col='star_rating', save_path=None):
        """
        Plot model performance across different segments.
        
        Args:
            df: DataFrame containing segment column
            y_true: True labels
            y_proba: Predicted probabilities
            segment_col: Column to segment by
            save_path: Optional path to save the plot
        """
        segments = df[segment_col].unique()
        metrics = []
        
        for segment in segments:
            mask = df[segment_col] == segment
            auc = roc_auc_score(y_true[mask], y_proba[mask])
            ap = average_precision_score(y_true[mask], y_proba[mask])
            metrics.append({'segment': segment, 'ROC-AUC': auc, 'PR-AUC': ap})
        
        metrics_df = pd.DataFrame(metrics).set_index('segment')
        
        plt.figure(figsize=(10, 5))
        metrics_df.plot(kind='bar', rot=0)
        plt.title(f"Performance by {segment_col}")
        plt.ylabel("Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_shap_importance(pipeline, X_test,y_test, feature_names=None, save_path=None):
        """
        Robust SHAP analysis implementation that handles:
        - Feature alignment issues
        - GradientBoostingClassifier quirks
        - Multiple fallback options
        """
        try:
            logger.info("Starting SHAP analysis...")
            
            # 1. Verify feature alignment
            expected_features = pipeline.named_steps['preprocessor'].feature_names_in_
            if len(X_test.columns) != len(expected_features):
                logger.error(f"Feature mismatch: Expected {len(expected_features)} features, got {len(X_test.columns)}")
                raise ValueError("Feature count mismatch between pipeline and test data")
            
            # 2. Prepare data through pipeline steps
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['classifier'].estimator
            
            try:
                X_test_transformed = preprocessor.transform(X_test)
            except ValueError as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                logger.info("Checking for feature name mismatches...")
                missing = set(expected_features) - set(X_test.columns)
                extra = set(X_test.columns) - set(expected_features)
                if missing:
                    logger.error(f"Missing features: {missing}")
                if extra:
                    logger.error(f"Extra features: {extra}")
                raise
            
            # 3. Get feature names after preprocessing
            if feature_names is None:
                num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                feature_names = np.concatenate([num_features, cat_features])
            
            # 4. SHAP analysis with multiple fallback options
            try:
                # Option 1: TreeExplainer with probability output
                logger.info("Attempting TreeExplainer...")
                explainer = shap.TreeExplainer(model, data=X_test_transformed, model_output='probability')
                shap_values = explainer.shap_values(X_test_transformed)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                    
            except Exception as tree_ex:
                logger.warning(f"TreeExplainer failed: {str(tree_ex)}")
                
                try:
                    # Option 2: KernelExplainer with background samples
                    logger.info("Falling back to KernelExplainer...")
                    background = shap.sample(X_test_transformed, min(100, len(X_test_transformed)))
                    
                    def predict_proba(X):
                        return pipeline.predict_proba(X)[:, 1]
                    
                    explainer = shap.KernelExplainer(predict_proba, background)
                    shap_values = explainer.shap_values(X_test_transformed)
                    
                except Exception as kernel_ex:
                    logger.warning(f"KernelExplainer failed: {str(kernel_ex)}")
                    
                    # Option 3: Permutation importance as last resort
                    logger.info("Falling back to permutation importance...")
                    from sklearn.inspection import permutation_importance
                    
                    result = permutation_importance(
                        pipeline, X_test, y_test, n_repeats=5, random_state=42
                    )
                    
                    # Create SHAP-like output
                    shap_values = np.tile(result.importances_mean, (X_test.shape[0], 1))
                    logger.warning("Using permutation importance instead of SHAP values")
            
            # 5. Generate and save plots
            plt.figure(figsize=(12, 8))
            
            if len(feature_names) == shap_values.shape[1]:  # Only plot if dimensions match
                shap.summary_plot(shap_values, X_test_transformed, 
                                feature_names=feature_names,
                                show=False)
                plt.title("SHAP Value Summary")
                
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    logger.info(f"SHAP plot saved to {save_path}")
                else:
                    plt.show()
            else:
                logger.error(f"Feature dimension mismatch: {len(feature_names)} names vs {shap_values.shape[1]} values")
            
            logger.info("SHAP analysis completed")
            
        except Exception as e:
            logger.error(f"SHAP analysis failed completely: {str(e)}")
            raise