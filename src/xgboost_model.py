import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from typing import Dict, List, Tuple

class XGBoostChallenger:
    def __init__(self, target_col: str = 'target'):
        self.target_col = target_col
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.features = []

    def fit_and_tune(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
        """
        Train and tune an XGBoost model using GridSearchCV.
        """
        self.features = X_train.columns.tolist()
        
        # Base classifier
        # Use a sensible scale_pos_weight based on class imbalance (e.g. total_goods / total_bads)
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos = num_neg / num_pos if num_pos > 0 else 1.0
        
        xgb_base = xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False,
            scale_pos_weight=scale_pos
        )
        
        # Small grid search for speed and robust tuning
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 150]
        }
        
        # Fit GridSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"XGBoost Best Parameters: {grid_search.best_params_}")
        return self.model

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default (PD) using XGBoost model.
        """
        # Ensure column order matches training
        return self.model.predict_proba(X[self.features])[:, 1]

    def run_shap_analysis(self, X: pd.DataFrame, output_html_path: str, output_img_path: str):
        """
        Compute SHAP values and save summary plots (HTML and PNG).
        """
        # Sample data if too large to speed up SHAP
        if len(X) > 1000:
            X_sample = X.sample(1000, random_state=42)
        else:
            X_sample = X.copy()
            
        # Fit SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(X_sample)
        
        # 1. Save summary plot as PNG
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X_sample, show=False)
        plt.title("XGBoost Challenger Model - SHAP Summary Plot", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=300)
        plt.close()
        
        # 2. Save interactive SHAP force plot or summary plot to HTML
        # Using SHAP's HTML output helper
        shap.plots.initjs()
        # For simplicity and robust display in html: we can render a force plot for the first 100 instances
        force_plot = shap.plots.force(self.explainer.expected_value, self.shap_values.values[:100], X_sample.iloc[:100])
        shap.save_html(output_html_path, force_plot)
        print(f"SHAP analysis saved to {output_html_path} and {output_img_path}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get standard feature importance from XGBoost.
        """
        importances = self.model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": self.features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        return imp_df
