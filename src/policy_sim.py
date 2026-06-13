import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

class PolicySimulatorPlotter:
    @staticmethod
    def plot_tradeoff_curves(simulation_df: pd.DataFrame, save_path: str = None):
        """
        Plot Approval Rate vs Expected Bad Rate Tradeoff Curve.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Score Cutoff')
        ax1.set_ylabel('Approval Rate (%)', color=color)
        line1 = ax1.plot(simulation_df['cutoff'], simulation_df['approval_rate_pct'], color=color, linewidth=2, label='Approval Rate (%)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Expected Portfolio Bad Rate (%)', color=color)
        line2 = ax2.plot(simulation_df['cutoff'], simulation_df['actual_bad_rate_pct'], color=color, linewidth=2, linestyle='--', label='Expected Bad Rate (%)')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # added these lines to combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title('Lending Policy Tradeoff: Approval Rate vs Expected Bad Rate', fontsize=12, pad=15)
        fig.tight_layout()  
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_profit_optimization(simulation_df: pd.DataFrame, save_path: str = None):
        """
        Plot Net Profit vs Cutoff Score.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(simulation_df['cutoff'], simulation_df['net_profit'] / 1e6, color='green', linewidth=2.5, label='Projected Net Profit ($M)')
        plt.axhline(0, color='red', linestyle=':', alpha=0.8)
        
        # Find max profit point
        max_idx = simulation_df['net_profit'].idxmax()
        max_profit_cutoff = simulation_df.loc[max_idx, 'cutoff']
        max_profit_val = simulation_df.loc[max_idx, 'net_profit'] / 1e6
        
        plt.scatter(max_profit_cutoff, max_profit_val, color='darkgreen', s=100, zorder=5)
        plt.annotate(f'Optimal Cutoff: {max_profit_cutoff}\nMax Profit: ${max_profit_val:.2f}M', 
                     xy=(max_profit_cutoff, max_profit_val), 
                     xytext=(max_profit_cutoff - 40, max_profit_val - 1.5),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                     fontsize=10, fontweight='bold')
                     
        plt.xlabel('Score Cutoff')
        plt.ylabel('Projected Net Profit ($ Millions)')
        plt.title('Lending Policy Profit Optimization', fontsize=12, pad=15)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower left')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
