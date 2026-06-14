import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Ensure src and root are in python path
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('.'))

from src.ecl_engine import ECLEngine

class TestECLEngine(unittest.TestCase):
    def setUp(self):
        # We will use patches to mock the loading of models during initialization
        self.patcher_pd = patch('src.ecl_engine.ScoreGenerationEngine')
        self.patcher_lgd = patch('src.ecl_engine.LGDModel')
        self.patcher_ead = patch('src.ecl_engine.EADModel')
        
        self.mock_pd_class = self.patcher_pd.start()
        self.mock_lgd_class = self.patcher_lgd.start()
        self.mock_ead_class = self.patcher_ead.start()
        
        # Setup mocks behavior
        self.mock_pd_instance = MagicMock()
        self.mock_pd_class.return_value = self.mock_pd_instance
        
        self.mock_lgd_instance = MagicMock()
        self.mock_lgd_class.return_value = self.mock_lgd_instance
        
        self.mock_ead_instance = MagicMock()
        self.mock_ead_class.return_value = self.mock_ead_instance
        
        # Instantiate engine
        self.engine = ECLEngine(
            scorecard_csv_path="mock_scorecard.csv",
            lgd_model_path="mock_lgd.pkl",
            ead_model_path="mock_ead.pkl",
            config_path="mock_config.yaml"
        )

    def tearDown(self):
        self.patcher_pd.stop()
        self.patcher_lgd.stop()
        self.patcher_ead.stop()

    def test_calculate_ecl_risk_tier(self):
        # ECL <= 1.5% -> Low ECL
        self.assertEqual(self.engine.calculate_ecl_risk_tier(0.010), "Low ECL")
        self.assertEqual(self.engine.calculate_ecl_risk_tier(0.015), "Low ECL")
        
        # 1.5% < ECL <= 5.0% -> Medium ECL
        self.assertEqual(self.engine.calculate_ecl_risk_tier(0.016), "Medium ECL")
        self.assertEqual(self.engine.calculate_ecl_risk_tier(0.050), "Medium ECL")
        
        # ECL > 5.0% -> High ECL
        self.assertEqual(self.engine.calculate_ecl_risk_tier(0.051), "High ECL")
        self.assertEqual(self.engine.calculate_ecl_risk_tier(0.100), "High ECL")

    def test_predict_applicant_ecl(self):
        # Configure Mock Outputs
        self.mock_pd_instance.score_applicant.return_value = {
            "credit_score": 710,
            "probability_of_default": 0.05,
            "risk_band": "Good",
            "decision": "Approved",
            "score_breakdown": {"annual_inc": 60},
            "bin_breakdown": {"annual_inc": "(30000.0, 60000.0]"}
        }
        
        # Mock LGD predictor to return 40% loss severity
        self.mock_lgd_instance.predict_lgd.return_value = np.array([0.40])
        
        # Mock EAD predictor to return 80% exposure percentage
        self.mock_ead_instance.predict_ead.return_value = np.array([0.80])
        
        # Test applicant data
        applicant = {
            "loan_amnt": 10000.0,
            "funded_amnt": 10000.0,
            "annual_inc": 50000.0,
            "dti": 15.0
        }
        
        res = self.engine.predict_applicant_ecl(applicant)
        
        # Check assertions
        # Expected PD = 0.05
        # Expected LGD = 0.40
        # Expected EAD% = 0.80 -> EAD Amount = 0.80 * 10000 = 8000
        # Expected ECL = 0.05 * 0.40 * 8000 = 160
        # Expected ECL% = 160 / 10000 = 0.016 (1.6%) -> Medium ECL
        self.assertEqual(res["credit_score"], 710)
        self.assertAlmostEqual(res["probability_of_default"], 0.05)
        self.assertAlmostEqual(res["lgd_percentage"], 0.40)
        self.assertAlmostEqual(res["ead_percentage"], 0.80)
        self.assertAlmostEqual(res["ead_amount"], 8000.0)
        self.assertAlmostEqual(res["expected_credit_loss"], 160.0)
        self.assertAlmostEqual(res["expected_credit_loss_percentage"], 0.016)
        self.assertEqual(res["ecl_risk_tier"], "Medium ECL")
        self.assertEqual(res["pd_risk_band"], "Good")
        self.assertEqual(res["pd_decision"], "Approved")

    def test_predict_batch_ecl(self):
        # Mock score_batch
        self.mock_pd_instance.score_batch.return_value = pd.DataFrame({
            "credit_score": [700, 600],
            "pd": [0.06, 0.15],
            "risk_band": ["Good", "High Risk"],
            "decision": ["Approved", "Refer to Underwriting"]
        })
        
        self.mock_lgd_instance.predict_lgd.return_value = np.array([0.50, 0.60])
        self.mock_ead_instance.predict_ead.return_value = np.array([0.90, 0.85])
        
        df = pd.DataFrame([
            {"loan_amnt": 10000.0, "funded_amnt": 10000.0, "annual_inc": 60000.0},
            {"loan_amnt": 20000.0, "funded_amnt": 20000.0, "annual_inc": 25000.0}
        ])
        
        res_df = self.engine.predict_batch_ecl(df)
        
        # Verify output dataframe columns and calculations
        self.assertIn("ecl", res_df.columns)
        self.assertIn("ecl_pct", res_df.columns)
        self.assertIn("ecl_risk_tier", res_df.columns)
        
        # Row 1: PD=0.06, LGD=0.50, EAD%=0.90 -> EAD Amt = 9000, ECL = 0.06 * 0.50 * 9000 = 270. ECL% = 2.7% -> Medium ECL
        # Row 2: PD=0.15, LGD=0.60, EAD%=0.85 -> EAD Amt = 17000, ECL = 0.15 * 0.60 * 17000 = 1530. ECL% = 7.65% -> High ECL
        self.assertAlmostEqual(res_df.loc[0, "ecl"], 270.0)
        self.assertEqual(res_df.loc[0, "ecl_risk_tier"], "Medium ECL")
        self.assertAlmostEqual(res_df.loc[1, "ecl"], 1530.0)
        self.assertEqual(res_df.loc[1, "ecl_risk_tier"], "High ECL")

if __name__ == '__main__':
    unittest.main()
