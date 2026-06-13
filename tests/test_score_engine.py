import unittest
import numpy as np
import pandas as pd
import os
from score_generation_engine import ScoreGenerationEngine

class TestScoreGenerationEngine(unittest.TestCase):
    def setUp(self):
        # Create a mock scorecard file for testing purposes
        self.mock_scorecard_path = "tests/mock_scorecard.csv"
        
        # Define some mock scorecard points
        mock_data = pd.DataFrame([
            {"Variable": "annual_inc", "Bin": "<= 30000.0", "WoE": -0.5, "Coefficient": -0.8, "Points": 45},
            {"Variable": "annual_inc", "Bin": "(30000.0, 60000.0]", "WoE": 0.1, "Coefficient": -0.8, "Points": 60},
            {"Variable": "annual_inc", "Bin": "> 60000.0", "WoE": 0.6, "Coefficient": -0.8, "Points": 75},
            {"Variable": "annual_inc", "Bin": "Missing", "WoE": 0.0, "Coefficient": -0.8, "Points": 55},
            
            {"Variable": "dti", "Bin": "<= 15.0", "WoE": 0.4, "Coefficient": -0.6, "Points": 70},
            {"Variable": "dti", "Bin": "(15.0, 30.0]", "WoE": -0.1, "Coefficient": -0.6, "Points": 50},
            {"Variable": "dti", "Bin": "> 30.0", "WoE": -0.6, "Coefficient": -0.6, "Points": 30},
            {"Variable": "dti", "Bin": "Missing", "WoE": 0.0, "Coefficient": -0.6, "Points": 55},
            
            {"Variable": "home_ownership", "Bin": "MORTGAGE, OWN", "WoE": 0.3, "Coefficient": -0.5, "Points": 65},
            {"Variable": "home_ownership", "Bin": "RENT, OTHER", "WoE": -0.3, "Coefficient": -0.5, "Points": 45},
            {"Variable": "home_ownership", "Bin": "Missing", "WoE": 0.0, "Coefficient": -0.5, "Points": 50}
        ])
        
        os.makedirs(os.path.dirname(self.mock_scorecard_path), exist_ok=True)
        mock_data.to_csv(self.mock_scorecard_path, index=False)
        
        # Load engine with mock scorecard
        self.engine = ScoreGenerationEngine(scorecard_csv_path=self.mock_scorecard_path)

    def tearDown(self):
        if os.path.exists(self.mock_scorecard_path):
            os.remove(self.mock_scorecard_path)

    def test_bin_parsing(self):
        # Numeric `<= 30000.0`
        left, right, bin_type = self.engine.parse_numeric_bin("<= 30000.0")
        self.assertEqual(left, -float('inf'))
        self.assertEqual(right, 30000.0)
        self.assertEqual(bin_type, "numeric")
        
        # Numeric `(30000.0, 60000.0]`
        left, right, bin_type = self.engine.parse_numeric_bin("(30000.0, 60000.0]")
        self.assertEqual(left, 30000.0)
        self.assertEqual(right, 60000.0)
        self.assertEqual(bin_type, "numeric")
        
        # Numeric `> 60000.0`
        left, right, bin_type = self.engine.parse_numeric_bin("> 60000.0")
        self.assertEqual(left, 60000.0)
        self.assertEqual(right, float('inf'))
        self.assertEqual(bin_type, "numeric")
        
        # Categorical `MORTGAGE, OWN`
        left, right, bin_type = self.engine.parse_numeric_bin("MORTGAGE, OWN")
        self.assertEqual(bin_type, "categorical")
        
        # Missing
        left, right, bin_type = self.engine.parse_numeric_bin("Missing")
        self.assertEqual(bin_type, "Missing")

    def test_lookup_points_numeric(self):
        # annual_inc <= 30k should return 45 points
        pts = self.engine.lookup_points("annual_inc", 25000)
        self.assertEqual(pts, 45)
        
        # annual_inc in (30k, 60k] should return 60 points
        pts = self.engine.lookup_points("annual_inc", 45000)
        self.assertEqual(pts, 60)
        
        # annual_inc > 60k should return 75 points
        pts = self.engine.lookup_points("annual_inc", 70000)
        self.assertEqual(pts, 75)
        
        # annual_inc missing should return 55 points
        pts = self.engine.lookup_points("annual_inc", np.nan)
        self.assertEqual(pts, 55)

    def test_lookup_points_categorical(self):
        # MORTGAGE should return 65 points
        pts = self.engine.lookup_points("home_ownership", "MORTGAGE")
        self.assertEqual(pts, 65)
        
        # RENT should return 45 points
        pts = self.engine.lookup_points("home_ownership", "RENT")
        self.assertEqual(pts, 45)
        
        # Missing category should return 50 points
        pts = self.engine.lookup_points("home_ownership", None)
        self.assertEqual(pts, 50)

    def test_score_calculation(self):
        applicant = {
            "annual_inc": 50000,       # (30k, 60k] -> 60 points
            "dti": 12.0,               # <= 15.0 -> 70 points
            "home_ownership": "MORTGAGE" # MORTGAGE, OWN -> 65 points
        }
        res = self.engine.score_applicant(applicant)
        
        # Total points should be: 60 + 70 + 65 = 195
        # Note: the mock values are scaled down. The scoring engine will clip it to min_score (300)
        # because the score 195 is below min_score. Let's verify it clips.
        self.assertEqual(res['credit_score'], 300)
        
        # Let's adjust points to be within range for testing addition
        # If we raise points:
        self.engine.min_score = 100 # lower limit for test
        res = self.engine.score_applicant(applicant)
        self.assertEqual(res['credit_score'], 195)
        self.assertEqual(res['score_breakdown']['annual_inc'], 60)
        self.assertEqual(res['score_breakdown']['dti'], 70)
        self.assertEqual(res['score_breakdown']['home_ownership'], 65)
        self.assertEqual(res['bin_breakdown']['annual_inc'], '(30000.0, 60000.0]')
        self.assertEqual(res['bin_breakdown']['dti'], '<= 15.0')
        self.assertEqual(res['bin_breakdown']['home_ownership'], 'MORTGAGE, OWN')

if __name__ == '__main__':
    unittest.main()
