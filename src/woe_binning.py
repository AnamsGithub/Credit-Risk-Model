import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Tuple

class WoEBinning:
    def __init__(self, target_col: str = 'target'):
        self.target_col = target_col
        self.mappings = {}
        self.iv_report = []
        self.selected_features = []

    def _calculate_woe_iv(self, df: pd.DataFrame, col: str, bins: list, is_numeric: bool) -> Tuple[List[Dict], float]:
        """
        Calculate Good/Bad counts, WoE, and IV for a given feature and its bins.
        """
        total_goods = df[df[self.target_col] == 0].shape[0]
        total_bads = df[df[self.target_col] == 1].shape[0]
        
        # Ensure we don't divide by zero
        total_goods = total_goods if total_goods > 0 else 1
        total_bads = total_bads if total_bads > 0 else 1
        
        bin_records = []
        iv_total = 0.0
        
        # We also need to calculate stats for missing values
        missing_mask = df[col].isnull()
        has_missing = missing_mask.sum() > 0
        
        # Process regular bins
        for idx, b in enumerate(bins):
            if is_numeric:
                # b is a tuple of (min_val, max_val)
                # handle edges correctly
                left, right = b
                if idx == 0:
                    mask = (df[col] <= right) & ~missing_mask
                elif idx == len(bins) - 1:
                    mask = (df[col] > left) & ~missing_mask
                else:
                    mask = (df[col] > left) & (df[col] <= right) & ~missing_mask
                bin_name = f"({round(left, 2)}, {round(right, 2)}]"
                if left == -float('inf'):
                    bin_name = f"<= {round(right, 2)}"
                elif right == float('inf'):
                    bin_name = f"> {round(left, 2)}"
            else:
                # b is a list of categories
                mask = df[col].isin(b) & ~missing_mask
                bin_name = ", ".join(map(str, b))
                
            goods = df[mask & (df[self.target_col] == 0)].shape[0]
            bads = df[mask & (df[self.target_col] == 1)].shape[0]
            total = goods + bads
            
            # Adjust to avoid division by zero
            g_cnt = goods if goods > 0 else 0.5
            b_cnt = bads if bads > 0 else 0.5
            
            g_pct = g_cnt / total_goods
            b_pct = b_cnt / total_bads
            
            woe = np.log(g_pct / b_pct)
            iv = (g_pct - b_pct) * woe
            iv_total += iv
            
            bin_records.append({
                "bin_name": bin_name,
                "range": b if is_numeric else list(b),
                "total_count": int(total),
                "good_count": int(goods),
                "bad_count": int(bads),
                "good_pct": float(g_pct),
                "bad_pct": float(b_pct),
                "default_rate": float(bads / total) if total > 0 else 0.0,
                "woe": float(woe),
                "iv": float(iv)
            })
            
        # Process missing values if they exist or as a fallback
        missing_goods = df[missing_mask & (df[self.target_col] == 0)].shape[0]
        missing_bads = df[missing_mask & (df[self.target_col] == 1)].shape[0]
        missing_total = missing_goods + missing_bads
        
        m_g_cnt = missing_goods if missing_goods > 0 else 0.5
        m_b_cnt = missing_bads if missing_bads > 0 else 0.5
        
        m_g_pct = m_g_cnt / total_goods
        m_b_pct = m_b_cnt / total_bads
        
        missing_woe = np.log(m_g_pct / m_b_pct)
        missing_iv = (m_g_pct - m_b_pct) * missing_woe
        
        if has_missing or missing_total > 0:
            iv_total += missing_iv
            bin_records.append({
                "bin_name": "Missing",
                "range": "Missing",
                "total_count": int(missing_total),
                "good_count": int(missing_goods),
                "bad_count": int(missing_bads),
                "good_pct": float(m_g_pct),
                "bad_pct": float(m_b_pct),
                "default_rate": float(missing_bads / missing_total) if missing_total > 0 else 0.0,
                "woe": float(missing_woe),
                "iv": float(missing_iv)
            })
            
        return bin_records, iv_total

    def fit_numeric(self, df: pd.DataFrame, col: str, max_bins: int = 5, min_pct: float = 0.05) -> Dict:
        """
        Determine optimal binning boundaries using Decision Tree, then compute WoE and IV.
        """
        # Exclude missing values from boundary calculation
        non_missing = df[df[col].notnull()].copy()
        
        if len(non_missing) == 0:
            # All values are missing
            self.mappings[col] = {
                "type": "numeric",
                "bins": [],
                "missing_woe": 0.0,
                "iv": 0.0,
                "details": [{"bin_name": "Missing", "woe": 0.0, "iv": 0.0}]
            }
            return self.mappings[col]
            
        X = non_missing[[col]]
        y = non_missing[self.target_col]
        
        # Fit a simple Decision Tree Classifier to find optimal splits
        min_samples = int(len(df) * min_pct)
        tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=min_samples, random_state=42)
        tree.fit(X, y)
        
        # Get threshold values and sort
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
        thresholds = sorted(list(set(thresholds)))
        
        # Create bins: e.g. [-inf, t1], (t1, t2], (t2, inf]
        bins = []
        if len(thresholds) == 0:
            # Fallback to simple quantiles if tree finds no splits
            q_splits = sorted(list(set(non_missing[col].quantile([0.2, 0.4, 0.6, 0.8]).tolist())))
            thresholds = q_splits
            
        if len(thresholds) > 0:
            bins.append((-float('inf'), thresholds[0]))
            for i in range(len(thresholds) - 1):
                bins.append((thresholds[i], thresholds[i+1]))
            bins.append((thresholds[-1], float('inf')))
        else:
            bins.append((-float('inf'), float('inf')))
            
        bin_details, iv = self._calculate_woe_iv(df, col, bins, is_numeric=True)
        
        # Extract missing woe
        missing_woe = 0.0
        for b in bin_details:
            if b['bin_name'] == 'Missing':
                missing_woe = b['woe']
                
        self.mappings[col] = {
            "type": "numeric",
            "bins": [{"min": float(b[0]), "max": float(b[1]), "woe": float(d['woe'])} for b, d in zip(bins, bin_details) if d['bin_name'] != 'Missing'],
            "missing_woe": float(missing_woe),
            "iv": float(iv),
            "details": bin_details
        }
        return self.mappings[col]

    def fit_categorical(self, df: pd.DataFrame, col: str, groups: List[List] = None) -> Dict:
        """
        Compute WoE and IV for a categorical column, optional pre-defined groups.
        """
        # If no groups provided, treat each unique value as its own group
        if groups is None:
            unique_vals = df[df[col].notnull()][col].unique().tolist()
            groups = [[val] for val in unique_vals]
            
        bin_details, iv = self._calculate_woe_iv(df, col, groups, is_numeric=False)
        
        # Extract missing woe
        missing_woe = 0.0
        for b in bin_details:
            if b['bin_name'] == 'Missing':
                missing_woe = b['woe']
                
        self.mappings[col] = {
            "type": "categorical",
            "bins": [{"categories": list(map(str, g)), "woe": float(d['woe'])} for g, d in zip(groups, bin_details) if d['bin_name'] != 'Missing'],
            "missing_woe": float(missing_woe),
            "iv": float(iv),
            "details": bin_details
        }
        return self.mappings[col]

    def fit_all(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
        """
        Fit WoE mapping on all numeric and categorical columns.
        """
        print("Fitting Numeric Columns...")
        for col in numeric_cols:
            if col in df.columns:
                print(f"  Fitting {col}...")
                # Special rules for highly skewed discrete integer indicators
                if col in ['delinq_2yrs', 'pub_rec', 'pub_rec_bankruptcies']:
                    # Force custom bins: 0, 1, 2+
                    # We can use fit_numeric with custom bins by bypassing decision tree if needed
                    # Let's customize boundaries for these skew variables
                    unique_vals = sorted(df[col].dropna().unique())
                    if len(unique_vals) <= 3:
                        # simple splits
                        self.mappings[col] = self.fit_numeric(df, col, max_bins=3)
                    else:
                        # force specific thresholds: 0, 1+ or 0, 1, 2+
                        non_missing = df[df[col].notnull()].copy()
                        bins = [(-float('inf'), 0), (0, 1), (1, float('inf'))]
                        bin_details, iv = self._calculate_woe_iv(df, col, bins, is_numeric=True)
                        missing_woe = 0.0
                        for b in bin_details:
                            if b['bin_name'] == 'Missing':
                                missing_woe = b['woe']
                        self.mappings[col] = {
                            "type": "numeric",
                            "bins": [{"min": float(b[0]), "max": float(b[1]), "woe": float(d['woe'])} for b, d in zip(bins, bin_details) if d['bin_name'] != 'Missing'],
                            "missing_woe": float(missing_woe),
                            "iv": float(iv),
                            "details": bin_details
                        }
                elif col == 'inq_last_6mths':
                    bins = [(-float('inf'), 0), (0, 1), (1, 2), (2, float('inf'))]
                    bin_details, iv = self._calculate_woe_iv(df, col, bins, is_numeric=True)
                    missing_woe = 0.0
                    for b in bin_details:
                        if b['bin_name'] == 'Missing':
                            missing_woe = b['woe']
                    self.mappings[col] = {
                        "type": "numeric",
                        "bins": [{"min": float(b[0]), "max": float(b[1]), "woe": float(d['woe'])} for b, d in zip(bins, bin_details) if d['bin_name'] != 'Missing'],
                        "missing_woe": float(missing_woe),
                        "iv": float(iv),
                        "details": bin_details
                    }
                else:
                    self.mappings[col] = self.fit_numeric(df, col, max_bins=5)
                    
        print("Fitting Categorical Columns...")
        for col in categorical_cols:
            if col in df.columns:
                print(f"  Fitting {col}...")
                if col == 'home_ownership':
                    # Group MORTGAGE, OWN as higher credit strength and RENT, OTHER, NONE as lower
                    groups = [['MORTGAGE'], ['OWN'], ['RENT'], ['OTHER', 'NONE', 'ANY']]
                    self.mappings[col] = self.fit_categorical(df, col, groups=groups)
                elif col == 'emp_length':
                    # We parsed emp_length to numerical, so let's treat it as numeric in numerical columns list!
                    # But if we treat it as categorical:
                    pass
                elif col == 'purpose':
                    # Group based on risk profile
                    # Calculate default rates for purpose to group them logically
                    dr_by_pur = df.groupby(col)[self.target_col].mean().sort_values()
                    # Group into: Low Risk (<10%), Medium Risk (10%-15%), High Risk (>15%)
                    low_risk = dr_by_pur[dr_by_pur < 0.12].index.tolist()
                    med_risk = dr_by_pur[(dr_by_pur >= 0.12) & (dr_by_pur < 0.17)].index.tolist()
                    high_risk = dr_by_pur[dr_by_pur >= 0.17].index.tolist()
                    groups = []
                    if low_risk: groups.append(low_risk)
                    if med_risk: groups.append(med_risk)
                    if high_risk: groups.append(high_risk)
                    self.mappings[col] = self.fit_categorical(df, col, groups=groups)
                else:
                    self.mappings[col] = self.fit_categorical(df, col)

        # Generate IV Ranking Report
        self.iv_report = []
        for col, mapping in self.mappings.items():
            iv_val = mapping['iv']
            if iv_val < 0.02:
                status = "Useless (IV < 0.02)"
            elif iv_val < 0.10:
                status = "Weak (0.02 - 0.10)"
            elif iv_val < 0.30:
                status = "Medium (0.10 - 0.30)"
            elif iv_val < 0.50:
                status = "Strong (0.30 - 0.50)"
            else:
                status = "Suspicious / Potential Leakage (IV > 0.50)"
                
            self.iv_report.append({
                "feature": col,
                "information_value": iv_val,
                "strength": status
            })
            
        self.iv_report = sorted(self.iv_report, key=lambda x: x['information_value'], reverse=True)
        
        # Feature selection: Select features between 0.02 and 0.50
        # Exception: We can keep a feature slightly above 0.50 if it is interest rate (int_rate),
        # but int_rate is highly correlated with default because default rates are higher. Let's see if we exclude suspicious ones.
        # Typically, a scorecard uses features with 0.02 to 0.50 to avoid leakage.
        # Let's filter selected features.
        self.selected_features = [
            item['feature'] for item in self.iv_report 
            if 0.02 <= item['information_value'] <= 0.50
        ]
        
        # If no features are selected, keep at least top 5 medium/strong features
        if len(self.selected_features) == 0:
            self.selected_features = [item['feature'] for item in self.iv_report[:5]]

    def transform_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        Transform a single raw column to its WoE values.
        """
        mapping = self.mappings[col]
        col_type = mapping['type']
        missing_woe = mapping['missing_woe']
        
        result = pd.Series(index=df.index, dtype=float)
        
        # Default missing values to missing_woe
        result[df[col].isnull()] = missing_woe
        
        non_missing_mask = df[col].notnull()
        
        if col_type == 'numeric':
            bins = mapping['bins']
            for b in bins:
                left = b['min']
                right = b['max']
                woe = b['woe']
                
                # Check boundaries
                if left == -float('inf') or left == float('-inf'):
                    mask = non_missing_mask & (df[col] <= right)
                elif right == float('inf') or right == float('inf'):
                    mask = non_missing_mask & (df[col] > left)
                else:
                    mask = non_missing_mask & (df[col] > left) & (df[col] <= right)
                result[mask] = woe
        else:
            # Categorical
            bins = mapping['bins']
            for b in bins:
                cats = b['categories']
                woe = b['woe']
                # match category strings
                mask = non_missing_mask & df[col].astype(str).isin(cats)
                result[mask] = woe
                
        # Fill any remaining unmapped values with missing_woe as fallback
        result = result.fillna(missing_woe)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform all columns in DataFrame to their WoE values.
        """
        woe_df = pd.DataFrame(index=df.index)
        for col in self.mappings.keys():
            if col in df.columns:
                woe_df[f"{col}_woe"] = self.transform_col(df, col)
        if self.target_col in df.columns:
            woe_df[self.target_col] = df[self.target_col]
        return woe_df

    def save_mappings(self, path: str):
        """
        Save binned mappings to JSON for scoring engine.
        """
        with open(path, "w") as file:
            json.dump(self.mappings, file, indent=4)

    def export_reports(self, woe_csv_path: str, iv_csv_path: str, excel_dict_path: str):
        """
        Export WoE details, IV rank, and feature dictionary.
        """
        # 1. WoE CSV
        woe_list = []
        for col, mapping in self.mappings.items():
            for b in mapping['details']:
                woe_list.append({
                    "variable": col,
                    "bin_name": b['bin_name'],
                    "total_count": b['total_count'],
                    "good_count": b['good_count'],
                    "bad_count": b['bad_count'],
                    "default_rate": b['default_rate'],
                    "woe": b['woe'],
                    "iv": b['iv']
                })
        woe_df = pd.DataFrame(woe_list)
        woe_df.to_csv(woe_csv_path, index=False)
        
        # 2. IV CSV
        iv_df = pd.DataFrame(self.iv_report)
        iv_df.to_csv(iv_csv_path, index=False)
        
        # 3. Feature Dictionary Excel
        # Write descriptive Excel sheet
        dict_list = []
        for col, mapping in self.mappings.items():
            dict_list.append({
                "Variable Name": col,
                "Type": mapping['type'],
                "IV": mapping['iv'],
                "Selected in Model": "Yes" if col in self.selected_features else "No",
                "Description": self._get_feature_description(col)
            })
        dict_df = pd.DataFrame(dict_list)
        
        with pd.ExcelWriter(excel_dict_path, engine='openpyxl') as writer:
            dict_df.to_excel(writer, sheet_name='Feature Dictionary', index=False)
            iv_df.to_excel(writer, sheet_name='IV Rankings', index=False)
            woe_df.to_excel(writer, sheet_name='WoE Table', index=False)

    def _get_feature_description(self, col: str) -> str:
        descriptions = {
            "loan_amnt": "The listed amount of the loan applied for by the borrower.",
            "int_rate": "Interest rate on the loan.",
            "annual_inc": "The self-reported annual income provided by the borrower during registration.",
            "dti": "A ratio calculated using the borrower's total monthly debt payments divided by the monthly income.",
            "revol_util": "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.",
            "delinq_2yrs": "The number of 30+ days past-due delinquencies in the borrower's credit file for the past 2 years.",
            "inq_last_6mths": "The number of inquiries in the past 6 months (excluding auto and home inquiries).",
            "open_acc": "The number of open credit lines in the borrower's credit file.",
            "pub_rec": "Number of derogatory public records.",
            "pub_rec_bankruptcies": "Number of public record bankruptcies.",
            "credit_history_age": "The age of the borrower's credit history in months, calculated between earliest_cr_line and issue_d.",
            "home_ownership": "The home ownership status provided by the borrower during registration (MORTGAGE, OWN, RENT, OTHER).",
            "purpose": "A category provided by the borrower for the loan request.",
            "emp_length": "Employment length in years (0 to 10 where 10 means 10+ years)."
        }
        return descriptions.get(col, "No description available.")
