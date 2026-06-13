import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class StabilityMonitor:
    @staticmethod
    def calculate_psi(expected_vals: np.ndarray, actual_vals: np.ndarray, num_buckets: int = 10, is_categorical: bool = False) -> Tuple[float, pd.DataFrame]:
        """
        Calculate Population Stability Index (PSI) between two arrays (Expected/Train vs Actual/OOT).
        """
        # Ensure we don't have empty arrays
        if len(expected_vals) == 0 or len(actual_vals) == 0:
            return 0.0, pd.DataFrame()
            
        if is_categorical:
            # Group by unique category
            expected_counts = pd.Series(expected_vals).value_counts()
            actual_counts = pd.Series(actual_vals).value_counts()
            
            # Combine all unique index values
            all_indices = sorted(list(set(expected_counts.index).union(set(actual_counts.index))))
            
            summary_df = pd.DataFrame(index=all_indices)
            summary_df['expected_count'] = summary_df.index.map(expected_counts).fillna(0)
            summary_df['actual_count'] = summary_df.index.map(actual_counts).fillna(0)
        else:
            # Bin numerical values using quantiles of expected values
            percentiles = np.linspace(0, 100, num_buckets + 1)
            # Handle duplicates by taking unique boundaries
            bins = np.percentile(expected_vals, percentiles)
            bins = np.unique(bins)
            # Adjust edges
            bins[0] = -float('inf')
            bins[-1] = float('inf')
            
            expected_cuts = pd.cut(expected_vals, bins=bins)
            actual_cuts = pd.cut(actual_vals, bins=bins)
            
            expected_counts = expected_cuts.value_counts().sort_index()
            actual_counts = actual_cuts.value_counts().sort_index()
            
            summary_df = pd.DataFrame({
                "expected_count": expected_counts,
                "actual_count": actual_counts
            })
            
        # Add small value (0.5) to avoid division by zero
        summary_df['expected_count_adj'] = summary_df['expected_count'].replace(0, 0.5)
        summary_df['actual_count_adj'] = summary_df['actual_count'].replace(0, 0.5)
        
        # Calculate percentages
        summary_df['expected_pct'] = summary_df['expected_count_adj'] / summary_df['expected_count_adj'].sum()
        summary_df['actual_pct'] = summary_df['actual_count_adj'] / summary_df['actual_count_adj'].sum()
        
        # PSI term
        summary_df['difference'] = summary_df['actual_pct'] - summary_df['expected_pct']
        summary_df['ln_ratio'] = np.log(summary_df['actual_pct'] / summary_df['expected_pct'])
        summary_df['psi_term'] = summary_df['difference'] * summary_df['ln_ratio']
        
        psi_value = float(summary_df['psi_term'].sum())
        
        # Clean up details for display
        summary_df.index.name = 'bin_range'
        summary_df = summary_df.reset_index()
        # Remove adjusted columns from output
        summary_df = summary_df[['bin_range', 'expected_count', 'actual_count', 'expected_pct', 'actual_pct', 'psi_term']]
        
        return psi_value, summary_df

    @staticmethod
    def calculate_csi(train_df: pd.DataFrame, oot_df: pd.DataFrame, woe_mappings: Dict) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
        """
        Calculate Characteristic Stability Index (CSI) for each characteristic/feature
        based on the WoE binning defined during training.
        """
        csi_values = {}
        csi_details = {}
        
        for col, mapping in woe_mappings.items():
            if col not in train_df.columns or col not in oot_df.columns:
                continue
                
            col_type = mapping['type']
            details = mapping['details']
            
            # Map values to their bin names
            # We can use our mapping logic to assign each record to a bin index/name
            def get_bin_name(val, map_details, type_col):
                if pd.isnull(val):
                    return "Missing"
                if type_col == 'numeric':
                    for d in map_details:
                        if d['bin_name'] == 'Missing':
                            continue
                        left, right = d['range']
                        if val > left and val <= right:
                            return d['bin_name']
                        # Special handling for outer edges
                        if left == -float('inf') and val <= right:
                            return d['bin_name']
                        if right == float('inf') and val > left:
                            return d['bin_name']
                else: # categorical
                    for d in map_details:
                        if d['bin_name'] == 'Missing':
                            continue
                        if str(val) in d['range']:
                            return d['bin_name']
                return "Missing" # fallback
                
            # Assign bins
            train_bins = train_df[col].apply(lambda x: get_bin_name(x, details, col_type))
            oot_bins = oot_df[col].apply(lambda x: get_bin_name(x, details, col_type))
            
            # Use categorical PSI to calculate CSI
            csi_val, csi_df = StabilityMonitor.calculate_psi(train_bins, oot_bins, is_categorical=True)
            
            csi_values[col] = csi_val
            csi_details[col] = csi_df
            
        return csi_values, csi_details

    @staticmethod
    def export_reports(psi_val: float, psi_details: pd.DataFrame, csi_values: Dict[str, float], csi_details: Dict[str, pd.DataFrame], psi_path: str, csi_path: str):
        """
        Export PSI and CSI detailed reports to Excel.
        """
        # Export PSI
        psi_summary_df = pd.DataFrame([{
            "Metric": "Population Stability Index (PSI)",
            "Value": psi_val,
            "Interpretation": "Stable" if psi_val < 0.10 else ("Moderate Shift" if psi_val < 0.25 else "Significant Drift")
        }])
        
        with pd.ExcelWriter(psi_path, engine='openpyxl') as writer:
            psi_summary_df.to_excel(writer, sheet_name='PSI Summary', index=False)
            psi_details.to_excel(writer, sheet_name='PSI Details', index=False)
            
        # Export CSI
        csi_summary_list = []
        for col, val in csi_values.items():
            csi_summary_list.append({
                "Characteristic": col,
                "CSI": val,
                "Interpretation": "Stable" if val < 0.10 else ("Moderate Shift" if val < 0.25 else "Significant Drift")
            })
        csi_summary_df = pd.DataFrame(csi_summary_list).sort_values(by="CSI", ascending=False)
        
        with pd.ExcelWriter(csi_path, engine='openpyxl') as writer:
            csi_summary_df.to_excel(writer, sheet_name='CSI Summary', index=False)
            for col, df in csi_details.items():
                # sheet name length limit is 31 chars
                sheet_name = f"CSI_{col[:27]}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
