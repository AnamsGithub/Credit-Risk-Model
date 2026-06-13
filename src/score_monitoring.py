import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class ScoreMonitor:
    def __init__(self, min_score: int = 300, max_score: int = 850):
        self.min_score = min_score
        self.max_score = max_score

    def assign_risk_band(self, score: float) -> str:
        """
        Assign risk bands based on score ranges.
        """
        if score >= 750:
            return "1. Excellent"
        elif score >= 700:
            return "2. Good"
        elif score >= 650:
            return "3. Moderate"
        elif score >= 600:
            return "4. High Risk"
        else:
            return "5. Very High Risk"

    def analyze_score_distribution(self, train_scores: np.ndarray, oot_scores: np.ndarray) -> Dict:
        """
        Generate summary stats for scores.
        """
        stats = {
            "train_count": len(train_scores),
            "oot_count": len(oot_scores),
            "train_mean": float(np.mean(train_scores)),
            "oot_mean": float(np.mean(oot_scores)),
            "train_median": float(np.median(train_scores)),
            "oot_median": float(np.median(oot_scores)),
            "train_std": float(np.std(train_scores)),
            "oot_std": float(np.std(oot_scores)),
            "train_min": int(np.min(train_scores)),
            "oot_min": int(np.min(oot_scores)),
            "train_max": int(np.max(train_scores)),
            "oot_max": int(np.max(oot_scores))
        }
        return stats

    def create_migration_matrix(self, oot_df: pd.DataFrame, oot_scores: np.ndarray) -> pd.DataFrame:
        """
        Track portfolio risk band distribution across sub-quarters of the OOT population (2011).
        This simulates portfolio risk migration monitoring.
        """
        df = oot_df.copy()
        df['score'] = oot_scores
        df['risk_band'] = df['score'].apply(self.assign_risk_band)
        
        # Split 2011 into Q1/Q2 (early 2011) vs Q3/Q4 (late 2011)
        # Using issue_month to define quarters: Q1/Q2 (Jan-Jun), Q3/Q4 (Jul-Dec)
        df['cohort'] = df['issue_month'].apply(lambda m: 'H1_2011' if m <= 6 else 'H2_2011')
        
        # Calculate distribution in H1 vs H2
        cohort_dist = df.groupby(['cohort', 'risk_band']).size().unstack(fill_value=0)
        cohort_dist_pct = cohort_dist.div(cohort_dist.sum(axis=1), axis=0) * 100
        
        return cohort_dist_pct.round(2)

    def plot_score_comparison(self, train_scores: np.ndarray, oot_scores: np.ndarray, save_path: str = None):
        """
        Plot score distribution comparison (Train vs OOT).
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(train_scores, bins=30, color='blue', label='Train Population', kde=True, stat='density', alpha=0.4)
        sns.histplot(oot_scores, bins=30, color='orange', label='OOT Population (2011)', kde=True, stat='density', alpha=0.4)
        
        # Add mean lines
        plt.axvline(np.mean(train_scores), color='darkblue', linestyle='--', linewidth=2, label=f'Train Mean ({np.mean(train_scores):.1f})')
        plt.axvline(np.mean(oot_scores), color='darkorange', linestyle='--', linewidth=2, label=f'OOT Mean ({np.mean(oot_scores):.1f})')
        
        plt.title("Credit Score Distribution Comparison (In-Time Train vs Out-of-Time)", fontsize=12, pad=15)
        plt.xlabel("Credit Score")
        plt.ylabel("Density")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.6)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_risk_band_comparison(self, train_scores: np.ndarray, oot_scores: np.ndarray, save_path: str = None):
        """
        Plot risk band percentage comparison.
        """
        train_bands = pd.Series([self.assign_risk_band(s) for s in train_scores]).value_counts(normalize=True).sort_index() * 100
        oot_bands = pd.Series([self.assign_risk_band(s) for s in oot_scores]).value_counts(normalize=True).sort_index() * 100
        
        bands_df = pd.DataFrame({
            "Train (%)": train_bands,
            "OOT (%)": oot_bands
        }).fillna(0)
        
        ax = bands_df.plot(kind='bar', figsize=(10, 6), color=['navy', 'orange'], width=0.7)
        plt.title("Credit Risk Band Distribution Comparison", fontsize=12, pad=15)
        plt.xlabel("Risk Category")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Add values on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def export_reports(self, stats: Dict, migration_matrix: pd.DataFrame, xlsx_path: str):
        """
        Export score monitoring statistics and migration matrices to Excel.
        """
        stats_df = pd.DataFrame([{
            "Metric": k,
            "Value": v
        } for k, v in stats.items()])
        
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='Score Summary Stats', index=False)
            migration_matrix.to_excel(writer, sheet_name='Quarterly Migration Matrix')
        print(f"Score monitoring report exported to {xlsx_path}")
