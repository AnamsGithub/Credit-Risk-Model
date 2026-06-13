import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class BusinessDecisionEngine:
    def __init__(self, avg_interest_rate: float = 0.12, loss_given_default: float = 0.60, avg_loan_term_years: float = 3.0):
        self.avg_interest_rate = avg_interest_rate
        self.loss_given_default = loss_given_default
        self.avg_loan_term_years = avg_loan_term_years

    def assign_risk_band(self, score: float) -> str:
        if score >= 750:
            return "Excellent"
        elif score >= 700:
            return "Good"
        elif score >= 650:
            return "Moderate"
        elif score >= 600:
            return "High Risk"
        else:
            return "Very High Risk"

    def summarize_risk_bands(self, df_scores_pd: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize portfolio by risk bands.
        df_scores_pd must contain: 'loan_amnt', 'score', 'pd', 'target'
        """
        df = df_scores_pd.copy()
        df['risk_band'] = df['score'].apply(self.assign_risk_band)
        
        summary = df.groupby('risk_band').agg(
            total_loans=('score', 'count'),
            total_amount=('loan_amnt', 'sum'),
            actual_defaults=('target', 'sum'),
            mean_score=('score', 'mean'),
            mean_pd=('pd', 'mean')
        ).reset_index()
        
        # Calculate percentages
        summary['pct_loans'] = summary['total_loans'] / len(df) * 100
        summary['pct_amount'] = summary['total_amount'] / df['loan_amnt'].sum() * 100
        summary['actual_default_rate'] = summary['actual_defaults'] / summary['total_loans'] * 100
        
        # Financial projections
        # Expected Revenue = amount * rate * term
        summary['expected_revenue'] = summary['total_amount'] * self.avg_interest_rate * self.avg_loan_term_years
        # Expected Loss = amount * pd * LGD
        summary['expected_loss'] = summary['total_amount'] * (summary['mean_pd']) * self.loss_given_default
        # Net Profit
        summary['net_profit'] = summary['expected_revenue'] - summary['expected_loss']
        summary['net_margin_pct'] = (summary['net_profit'] / summary['total_amount']) * 100
        
        # Order risk bands logically
        band_order = {"Excellent": 0, "Good": 1, "Moderate": 2, "High Risk": 3, "Very High Risk": 4}
        summary['order'] = summary['risk_band'].map(band_order)
        summary = summary.sort_values(by='order').drop(columns=['order']).reset_index(drop=True)
        
        return summary

    def run_cutoff_simulation(self, df_scores_pd: pd.DataFrame, min_cutoff: int = 500, max_cutoff: int = 800, step: int = 10) -> pd.DataFrame:
        """
        Simulate portfolio metrics for different score cutoffs.
        """
        df = df_scores_pd.copy()
        cutoffs = list(range(min_cutoff, max_cutoff + step, step))
        results = []
        
        total_applicants = len(df)
        total_loan_amount = df['loan_amnt'].sum()
        
        for cutoff in cutoffs:
            approved = df[df['score'] >= cutoff]
            n_approved = len(approved)
            
            if n_approved == 0:
                results.append({
                    "cutoff": cutoff,
                    "approval_rate_pct": 0.0,
                    "approved_loans": 0,
                    "approved_amount": 0.0,
                    "portfolio_pd_pct": 0.0,
                    "actual_bad_rate_pct": 0.0,
                    "expected_revenue": 0.0,
                    "expected_loss": 0.0,
                    "net_profit": 0.0,
                    "net_margin_pct": 0.0
                })
                continue
                
            approved_amount = approved['loan_amnt'].sum()
            mean_pd = approved['pd'].mean()
            actual_bad_rate = approved['target'].mean()
            
            # Revenue = Approved Amount * Interest Rate * Term
            exp_rev = approved_amount * self.avg_interest_rate * self.avg_loan_term_years
            # Loss = Approved Amount * Mean PD * LGD
            exp_loss = approved_amount * mean_pd * self.loss_given_default
            net_profit = exp_rev - exp_loss
            
            results.append({
                "cutoff": cutoff,
                "approval_rate_pct": (n_approved / total_applicants) * 100,
                "approved_loans": n_approved,
                "approved_amount": float(approved_amount),
                "portfolio_pd_pct": float(mean_pd * 100),
                "actual_bad_rate_pct": float(actual_bad_rate * 100),
                "expected_revenue": float(exp_rev),
                "expected_loss": float(exp_loss),
                "net_profit": float(net_profit),
                "net_margin_pct": (net_profit / approved_amount) * 100 if approved_amount > 0 else 0.0
            })
            
        return pd.DataFrame(results)

    def recommend_optimal_cutoff(self, simulation_df: pd.DataFrame) -> Dict:
        """
        Recommend an optimal cutoff score based on profit maximization
        and default rate constraints (e.g. keeping default rate below 5%).
        """
        # Criteria 1: Max Net Profit
        max_profit_row = simulation_df.loc[simulation_df['net_profit'].idxmax()]
        
        # Criteria 2: Conservative (Bad rate < 10% and high approval rate)
        sub_df = simulation_df[simulation_df['actual_bad_rate_pct'] < 10.0]
        if len(sub_df) > 0:
            conservative_row = sub_df.loc[sub_df['approval_rate_pct'].idxmax()]
        else:
            conservative_row = max_profit_row
            
        # Criteria 3: Balanced (Bad rate < 12% and maximizing profit)
        sub_df_balanced = simulation_df[simulation_df['actual_bad_rate_pct'] < 12.0]
        if len(sub_df_balanced) > 0:
            balanced_row = sub_df_balanced.loc[sub_df_balanced['net_profit'].idxmax()]
        else:
            balanced_row = max_profit_row
            
        recommendation = {
            "profit_max": {
                "cutoff": int(max_profit_row['cutoff']),
                "approval_rate": float(max_profit_row['approval_rate_pct']),
                "bad_rate": float(max_profit_row['actual_bad_rate_pct']),
                "net_profit": float(max_profit_row['net_profit'])
            },
            "balanced": {
                "cutoff": int(balanced_row['cutoff']),
                "approval_rate": float(balanced_row['approval_rate_pct']),
                "bad_rate": float(balanced_row['actual_bad_rate_pct']),
                "net_profit": float(balanced_row['net_profit'])
            },
            "conservative": {
                "cutoff": int(conservative_row['cutoff']),
                "approval_rate": float(conservative_row['approval_rate_pct']),
                "bad_rate": float(conservative_row['actual_bad_rate_pct']),
                "net_profit": float(conservative_row['net_profit'])
            }
        }
        return recommendation
