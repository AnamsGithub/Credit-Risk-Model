import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Tuple

class ScorecardModel:
    def __init__(self, base_score: float = 600, base_odds: float = 50, pdo: float = 20, min_score: float = 300, max_score: float = 850):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.min_score = min_score
        self.max_score = max_score
        
        # Scaling Factor and Offset
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)
        
        self.features = []
        self.model = None
        self.intercept = 0.0
        self.coefficients = {}
        self.p_values = {}
        self.vif_report = pd.DataFrame()
        self.scorecard_table = None

    def check_multicollinearity(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF) for the features.
        """
        # Add a constant for VIF calculation
        X_const = sm.add_constant(X)
        vif_data = []
        # Index starts at 1 to skip the constant
        for i in range(1, X_const.shape[1]):
            col = X_const.columns[i]
            vif = variance_inflation_factor(X_const.values, i)
            vif_data.append({
                "feature": col.replace("_woe", ""),
                "VIF": vif
            })
        self.vif_report = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)
        return self.vif_report

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit Logistic Regression model using statsmodels and check statistical metrics.
        """
        self.features = X.columns.tolist()
        
        # Add constant for intercept
        X_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_const)
        self.model = logit_model.fit(disp=False)
        
        # Extract model coefficients
        params = self.model.params
        pvals = self.model.pvalues
        
        self.intercept = float(params['const'])
        for col in self.features:
            self.coefficients[col] = float(params[col])
            self.p_values[col] = float(pvals[col])
            
        # Run VIF check
        self.check_multicollinearity(X)

    def validate_coefficients(self) -> Tuple[bool, List[str]]:
        """
        Check that all coefficients are negative (since high WoE must reduce default log-odds)
        and statistically significant (p-value < 0.05).
        """
        issues = []
        is_valid = True
        
        for col in self.features:
            coef = self.coefficients[col]
            pval = self.p_values[col]
            
            # Check sign: WoE is log(Good%/Bad%). Higher WoE means lower risk.
            # In logistic regression predicting Y=1 (Default), the coefficient must be negative.
            if coef >= 0:
                is_valid = False
                issues.append(f"Feature '{col}' has a counter-intuitive positive coefficient ({round(coef, 4)}). This implies higher WoE increases default risk.")
                
            # Check statistical significance
            if pval >= 0.05:
                is_valid = False
                issues.append(f"Feature '{col}' is not statistically significant (p-value = {round(pval, 4)} >= 0.05).")
                
        # Check multicollinearity (VIF > 3.0 is a warning in scorecards)
        for idx, row in self.vif_report.iterrows():
            if row['VIF'] > 3.0:
                issues.append(f"Feature '{row['feature']}' has high multicollinearity (VIF = {round(row['VIF'], 2)} > 3.0).")
                
        return is_valid, issues

    def build_scorecard_table(self, woe_mappings: Dict) -> pd.DataFrame:
        """
        Convert the logistic regression model into a scorecard table with points per bin.
        """
        scorecard_rows = []
        n_features = len(self.features)
        
        for woe_col in self.features:
            raw_col = woe_col.replace("_woe", "")
            coef = self.coefficients[woe_col]
            mapping = woe_mappings[raw_col]
            
            # Negative coefficient in default model is a positive coefficient in Good Odds model.
            # Good Odds Coefficient: theta = -beta
            theta = -coef
            
            # Process regular bins
            for b in mapping['details']:
                bin_name = b['bin_name']
                woe_val = b['woe']
                
                # Formula: Points = (WoE * theta - alpha/N) * Factor + Offset/N
                # Note: alpha is intercept
                points = (woe_val * theta - self.intercept / n_features) * self.factor + (self.offset / n_features)
                
                scorecard_rows.append({
                    "Variable": raw_col,
                    "Bin": bin_name,
                    "WoE": woe_val,
                    "Coefficient": coef,
                    "Points": int(round(points))
                })
                
        self.scorecard_table = pd.DataFrame(scorecard_rows)
        return self.scorecard_table

    def predict_score(self, df_woe: pd.DataFrame) -> pd.Series:
        """
        Predict credit score directly using WoE-transformed features.
        """
        # Score = Sum of Points.
        # Mathematically: Score = ln(Odds) * Factor + Offset
        # ln(Odds) = -intercept - sum(coef_i * woe_i)
        
        # Calculate log-odds of good: ln(1-p/p)
        log_odds_good = -self.intercept
        for woe_col in self.features:
            log_odds_good -= self.coefficients[woe_col] * df_woe[woe_col]
            
        scores = log_odds_good * self.factor + self.offset
        
        # Clip scores to regulatory minimum and maximum
        return scores.clip(self.min_score, self.max_score).round().astype(int)

    def predict_pd(self, df_woe: pd.DataFrame) -> pd.Series:
        """
        Predict probability of default (PD) using model coefficients.
        """
        # ln(p / (1-p)) = intercept + sum(coef_i * woe_i)
        log_odds_default = self.intercept + df_woe[self.features].dot(pd.Series(self.coefficients))
        pd_series = 1 / (1 + np.exp(-log_odds_default))
        return pd_series

    def save_scorecard(self, path: str):
        if self.scorecard_table is not None:
            self.scorecard_table.to_csv(path, index=False)
