import json
import pandas as pd
import numpy as np
import os
import yaml
from typing import Dict, Union, List

class ScoreGenerationEngine:
    def __init__(self, scorecard_csv_path: str = "outputs/scorecards/scorecard_table.csv", config_path: str = "configs/config.yaml"):
        self.scorecard_csv_path = scorecard_csv_path
        self.config_path = config_path
        self.scorecard_df = None
        self.factor = 28.8539  # Fallback defaults if config missing
        self.offset = 487.123
        self.min_score = 300
        self.max_score = 850
        self.features = []
        
        self.load_scorecard()
        self.load_config()

    def load_config(self):
        """
        Load scaling parameters from config.yaml
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
                scaling = config.get("scaling", {})
                base_score = scaling.get("base_score", 600)
                base_odds = scaling.get("base_odds", 50)
                pdo = scaling.get("pdo", 20)
                self.min_score = scaling.get("minimum_score", 300)
                self.max_score = scaling.get("maximum_score", 850)
                
                # Re-calculate Factor and Offset
                self.factor = pdo / np.log(2)
                self.offset = base_score - self.factor * np.log(base_odds)

    def load_scorecard(self):
        """
        Load scorecard lookup table from CSV.
        """
        if not os.path.exists(self.scorecard_csv_path):
            raise FileNotFoundError(f"Scorecard table not found at {self.scorecard_csv_path}. Please run training first.")
        self.scorecard_df = pd.read_csv(self.scorecard_csv_path)
        self.features = self.scorecard_df['Variable'].unique().tolist()

    def parse_numeric_bin(self, bin_str: str) -> tuple:
        """
        Parse bin string like '(30000.0, 60000.0]' or '<= 30.0' or '> 60.0' into mathematical ranges.
        """
        bin_str = bin_str.strip()
        if bin_str == "Missing":
            return (None, None, "Missing")
            
        if bin_str.startswith("<="):
            val = float(bin_str.replace("<=", "").strip())
            return (-float('inf'), val, "numeric")
        elif bin_str.startswith(">"):
            val = float(bin_str.replace(">", "").strip())
            return (val, float('inf'), "numeric")
        elif bin_str.startswith("(") and bin_str.endswith("]"):
            parts = bin_str[1:-1].split(",")
            left = float(parts[0].strip())
            right = float(parts[1].strip())
            return (left, right, "numeric")
            
        return (None, None, "categorical")

    def lookup_points(self, variable: str, value: Union[int, float, str]) -> int:
        """
        Look up scorecard points for a specific variable and borrower value.
        """
        _, points = self.lookup_bin_and_points(variable, value)
        return points

    def lookup_bin_and_points(self, variable: str, value: Union[int, float, str]) -> tuple:
        """
        Look up both the matched bin name and the scorecard points for a specific variable and borrower value.
        """
        sub_df = self.scorecard_df[self.scorecard_df['Variable'] == variable]
        
        # 1. Handle missing values
        if pd.isnull(value) or str(value).strip().lower() in ['nan', 'null', 'missing', 'n/a']:
            missing_row = sub_df[sub_df['Bin'] == 'Missing']
            if len(missing_row) > 0:
                return 'Missing', int(missing_row['Points'].values[0])
            return 'Missing', 0
            
        # 2. Iterate through bins to find match
        for idx, row in sub_df.iterrows():
            bin_name = row['Bin']
            if bin_name == "Missing":
                continue
                
            left, right, bin_type = self.parse_numeric_bin(bin_name)
            
            if bin_type == "numeric":
                try:
                    num_val = float(value)
                    if left == -float('inf') and num_val <= right:
                        return bin_name, int(row['Points'])
                    elif right == float('inf') and num_val > left:
                        return bin_name, int(row['Points'])
                    elif left is not None and right is not None and num_val > left and num_val <= right:
                        return bin_name, int(row['Points'])
                except (ValueError, TypeError):
                    pass
            else:
                categories = [c.strip() for c in bin_name.split(",")]
                if str(value).strip() in categories:
                    return bin_name, int(row['Points'])
                    
        # Fallback to missing/default
        missing_row = sub_df[sub_df['Bin'] == 'Missing']
        if len(missing_row) > 0:
            return 'Missing', int(missing_row['Points'].values[0])
        return sub_df['Bin'].values[0] if len(sub_df) > 0 else 'Unknown', int(sub_df['Points'].values[0]) if len(sub_df) > 0 else 0

    def score_applicant(self, applicant_data: Dict) -> Dict:
        """
        Score a single applicant.
        applicant_data is a dict containing raw applicant features (e.g. {'annual_inc': 55000, 'dti': 18.5})
        """
        score_breakdown = {}
        bin_breakdown = {}
        total_score = 0
        
        for var in self.features:
            val = applicant_data.get(var, np.nan)
            matched_bin, points = self.lookup_bin_and_points(var, val)
            score_breakdown[var] = points
            bin_breakdown[var] = matched_bin
            total_score += points
            
        # Clip score
        final_score = int(np.clip(total_score, self.min_score, self.max_score))
        
        # Calculate Probability of Default (PD) from score
        # PD = 1 / (1 + exp((Score - Offset)/Factor))
        odds = np.exp((final_score - self.offset) / self.factor)
        pd_val = 1 / (1 + odds)
        
        # Risk band assignment
        if final_score >= 750:
            risk_band = "Excellent"
            decision = "Approved"
        elif final_score >= 700:
            risk_band = "Good"
            decision = "Approved"
        elif final_score >= 650:
            risk_band = "Moderate"
            decision = "Approved"
        elif final_score >= 600:
            risk_band = "High Risk"
            decision = "Refer to Underwriting"
        else:
            risk_band = "Very High Risk"
            decision = "Declined"
            
        return {
            "credit_score": final_score,
            "probability_of_default": float(pd_val),
            "risk_band": risk_band,
            "decision": decision,
            "score_breakdown": score_breakdown,
            "bin_breakdown": bin_breakdown
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch of applicants in a DataFrame.
        """
        scores = []
        pds = []
        bands = []
        decisions = []
        
        for idx, row in df.iterrows():
            res = self.score_applicant(row.to_dict())
            scores.append(res['credit_score'])
            pds.append(res['probability_of_default'])
            bands.append(res['risk_band'])
            decisions.append(res['decision'])
            
        res_df = df.copy()
        res_df['credit_score'] = scores
        res_df['pd'] = pds
        res_df['risk_band'] = bands
        res_df['decision'] = decisions
        return res_df
