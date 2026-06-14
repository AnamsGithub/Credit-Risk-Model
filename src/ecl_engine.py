import os
import numpy as np
import pandas as pd
from typing import Dict, Union
from score_generation_engine import ScoreGenerationEngine
from lgd_model import LGDModel
from ead_model import EADModel

class ECLEngine:
    def __init__(
        self,
        scorecard_csv_path: str = "outputs/scorecards/scorecard_table.csv",
        lgd_model_path: str = "outputs/scorecards/lgd_model.pkl",
        ead_model_path: str = "outputs/scorecards/ead_model.pkl",
        config_path: str = "configs/config.yaml"
    ):
        self.scorecard_csv_path = scorecard_csv_path
        self.lgd_model_path = lgd_model_path
        self.ead_model_path = ead_model_path
        self.config_path = config_path
        
        # Instantiate component models
        print("Initializing ECL Engine components...")
        self.pd_engine = ScoreGenerationEngine(
            scorecard_csv_path=self.scorecard_csv_path,
            config_path=self.config_path
        )
        
        self.lgd_model = LGDModel()
        if os.path.exists(self.lgd_model_path):
            self.lgd_model.load_model(self.lgd_model_path)
        else:
            print(f"Warning: LGD model file not found at {self.lgd_model_path}. Model will need to be fit.")
            
        self.ead_model = EADModel()
        if os.path.exists(self.ead_model_path):
            self.ead_model.load_model(self.ead_model_path)
        else:
            print(f"Warning: EAD model file not found at {self.ead_model_path}. Model will need to be fit.")

    def calculate_ecl_risk_tier(self, ecl_pct: float) -> str:
        """
        Assign Expected Credit Loss (ECL) Risk Tier based on ECL as a percentage of Loan Amount:
        - Low ECL: ECL <= 1.5%
        - Medium ECL: 1.5% < ECL <= 5.0%
        - High ECL: ECL > 5.0%
        """
        if ecl_pct <= 0.015:
            return "Low ECL"
        elif ecl_pct <= 0.050:
            return "Medium ECL"
        else:
            return "High ECL"

    def predict_applicant_ecl(self, applicant_data: dict) -> dict:
        """
        Calculate expected credit loss metrics for a single applicant.
        """
        # 1. Run Probability of Default (PD) Scoring Engine
        pd_result = self.pd_engine.score_applicant(applicant_data)
        pd_val = pd_result["probability_of_default"]
        credit_score = pd_result["credit_score"]
        
        # 2. Get Loan/Funded Amount
        loan_amount = float(applicant_data.get("loan_amnt", 0.0))
        # Default to loan amount if funded amount is not present
        funded_amount = float(applicant_data.get("funded_amnt", loan_amount))
        
        if loan_amount <= 0:
            # Avoid division by zero
            loan_amount = 1000.0
            funded_amount = 1000.0
            
        # 3. Predict LGD % (Loss Given Default)
        df_single = pd.DataFrame([applicant_data])
        # Add basic columns if missing so preprocess runs without error
        if "loan_amnt" not in df_single.columns:
            df_single["loan_amnt"] = loan_amount
            
        lgd_val = float(self.lgd_model.predict_lgd(df_single, model_type='xgb')[0])
        
        # 4. Predict EAD % (Exposure at Default)
        if "funded_amnt" not in df_single.columns:
            df_single["funded_amnt"] = funded_amount
            
        ead_pct = float(self.ead_model.predict_ead(df_single)[0])
        
        # 5. Calculate ECL Amounts
        ead_amount = ead_pct * funded_amount
        ecl_amount = pd_val * lgd_val * ead_amount
        ecl_pct = ecl_amount / funded_amount
        
        # 6. Assign Risk Tier
        risk_tier = self.calculate_ecl_risk_tier(ecl_pct)
        
        return {
            "credit_score": credit_score,
            "probability_of_default": pd_val,
            "lgd_percentage": lgd_val,
            "ead_percentage": ead_pct,
            "ead_amount": ead_amount,
            "expected_credit_loss": ecl_amount,
            "expected_credit_loss_percentage": ecl_pct,
            "ecl_risk_tier": risk_tier,
            "pd_risk_band": pd_result["risk_band"],
            "pd_decision": pd_result["decision"],
            "score_breakdown": pd_result["score_breakdown"],
            "bin_breakdown": pd_result["bin_breakdown"]
        }

    def predict_batch_ecl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Expected Credit Loss metrics for a batch of applicants in a DataFrame.
        """
        res_df = df.copy()
        
        # Get PDs and scores
        res_pd = self.pd_engine.score_batch(df)
        res_df["credit_score"] = res_pd["credit_score"]
        res_df["pd"] = res_pd["pd"]
        res_df["pd_risk_band"] = res_pd["risk_band"]
        res_df["pd_decision"] = res_pd["decision"]
        
        # Predict LGD% and EAD%
        res_df["lgd"] = self.lgd_model.predict_lgd(df, model_type='xgb')
        res_df["ead_pct"] = self.ead_model.predict_ead(df)
        
        # Exposure amounts
        loan_amount = res_df["loan_amnt"].fillna(1000.0)
        funded_amount = res_df["funded_amnt"].fillna(loan_amount)
        
        res_df["ead_amount"] = res_df["ead_pct"] * funded_amount
        res_df["ecl"] = res_df["pd"] * res_df["lgd"] * res_df["ead_amount"]
        res_df["ecl_pct"] = res_df["ecl"] / funded_amount.replace(0.0, 1.0)
        
        # Apply Risk Tier mapping
        res_df["ecl_risk_tier"] = res_df["ecl_pct"].apply(self.calculate_ecl_risk_tier)
        
        return res_df
