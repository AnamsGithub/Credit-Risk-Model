import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class ModelValidator:
    @staticmethod
    def calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate KS statistic and return necessary arrays for plotting.
        """
        # Create a DataFrame
        df = pd.DataFrame({'target': y_true, 'probability': y_prob})
        df = df.sort_values(by='probability', ascending=True).reset_index(drop=True)
        
        # Calculate Good and Bad counts
        df['good'] = 1 - df['target']
        df['bad'] = df['target']
        
        df['cum_good_pct'] = df['good'].cumsum() / df['good'].sum()
        df['cum_bad_pct'] = df['bad'].cumsum() / df['bad'].sum()
        
        # KS separation
        df['ks_diff'] = np.abs(df['cum_good_pct'] - df['cum_bad_pct'])
        ks_stat = df['ks_diff'].max()
        
        # Find index/probability at maximum separation
        max_idx = df['ks_diff'].idxmax()
        ks_threshold = df.loc[max_idx, 'probability']
        
        return float(ks_stat), float(ks_threshold), df['probability'].values, df['cum_good_pct'].values, df['cum_bad_pct'].values

    @staticmethod
    def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Gini = 2 * AUC - 1
        """
        auc = roc_auc_score(y_true, y_prob)
        return float(2 * auc - 1)

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Brier Score (Mean Squared Error of probabilities)
        """
        return float(brier_score_loss(y_true, y_prob))

    @staticmethod
    def generate_decile_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        """
        Create decile-level summary table based on risk rankings.
        """
        df = pd.DataFrame({'target': y_true, 'probability': y_prob})
        
        # Sort in descending order of risk (highest PD first)
        df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
        
        # Cut into 10 groups (deciles)
        df['decile'] = pd.qcut(df.index, 10, labels=False, duplicates='drop') + 1
        
        decile_summary = df.groupby('decile').agg(
            total_loans=('target', 'count'),
            defaults=('target', 'sum'),
            min_pd=('probability', 'min'),
            max_pd=('probability', 'max'),
            mean_pd=('probability', 'mean')
        ).reset_index()
        
        decile_summary['non_defaults'] = decile_summary['total_loans'] - decile_summary['defaults']
        decile_summary['default_rate'] = decile_summary['defaults'] / decile_summary['total_loans']
        
        # Cumulative metrics
        total_defaults = decile_summary['defaults'].sum()
        total_non_defaults = decile_summary['non_defaults'].sum()
        
        decile_summary['cum_defaults'] = decile_summary['defaults'].cumsum()
        decile_summary['cum_non_defaults'] = decile_summary['non_defaults'].cumsum()
        
        decile_summary['cum_defaults_pct'] = decile_summary['cum_defaults'] / total_defaults
        decile_summary['cum_non_defaults_pct'] = decile_summary['cum_non_defaults'] / total_non_defaults
        
        decile_summary['ks_stat'] = np.abs(decile_summary['cum_defaults_pct'] - decile_summary['cum_non_defaults_pct'])
        decile_summary['lift'] = (decile_summary['defaults'] / decile_summary['total_loans']) / (total_defaults / len(y_true))
        
        return decile_summary

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str = None):
        """
        Plot and save ROC curve.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(title, fontsize=12, pad=15)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_ks_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str = None):
        """
        Plot and save KS separation curve.
        """
        ks_stat, ks_thresh, probs, cum_good, cum_bad = ModelValidator.calculate_ks(y_true, y_prob)
        
        plt.figure(figsize=(7, 6))
        plt.plot(probs, cum_bad, color='red', lw=2, label='Cumulative Default (Bad)')
        plt.plot(probs, cum_good, color='green', lw=2, label='Cumulative Non-Default (Good)')
        
        # Draw vertical line at max separation
        plt.axvline(x=ks_thresh, color='blue', linestyle='--', label=f'KS Stat = {ks_stat:.3f} (PD={ks_thresh:.3f})')
        
        plt.xlabel('Probability of Default (PD)')
        plt.ylabel('Cumulative Percentage')
        plt.title(title, fontsize=12, pad=15)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, save_path: str = None):
        """
        Plot and save calibration curve.
        """
        df = pd.DataFrame({'target': y_true, 'probability': y_prob})
        df['bin'] = pd.qcut(df['probability'], 10, labels=False, duplicates='drop')
        
        calib = df.groupby('bin').agg(
            mean_pred=('probability', 'mean'),
            actual_rate=('target', 'mean')
        ).reset_index()
        
        plt.figure(figsize=(7, 6))
        plt.plot(calib['mean_pred'], calib['actual_rate'], marker='o', linewidth=2, color='purple', label='Model Calibration')
        plt.plot([0, max(calib['mean_pred'].max(), calib['actual_rate'].max())], 
                 [0, max(calib['mean_pred'].max(), calib['actual_rate'].max())], 
                 linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability of Default (PD)')
        plt.ylabel('Actual Default Rate')
        plt.title(title, fontsize=12, pad=15)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.6)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
