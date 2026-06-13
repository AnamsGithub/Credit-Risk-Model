import pandas as pd
import numpy as np
from typing import Tuple

class DataProcessor:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.raw_df = None
        self.cleaned_df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load Lending Club dataset and perform basic checks.
        """
        self.raw_df = pd.read_csv(self.raw_data_path, low_memory=False)
        return self.raw_df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean raw fields, handle missing values, and calculate credit features.
        """
        if self.raw_df is None:
            self.load_data()
            
        df = self.raw_df.copy()
        
        # 1. Clean Percentage fields (remove '%', convert to float)
        if 'int_rate' in df.columns:
            df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').str.strip()
            df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')
            
        if 'revol_util' in df.columns:
            df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '').str.strip()
            df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')
            
        # 2. Clean Employment Length (convert to numeric ordinal, handle '10+ years' and '< 1 year')
        if 'emp_length' in df.columns:
            df['emp_length_raw'] = df['emp_length'].copy()
            df['emp_length'] = df['emp_length'].astype(str).str.strip()
            # Map strings to numeric values
            emp_map = {
                '< 1 year': 0,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10,
                'n/a': np.nan,
                'nan': np.nan
            }
            df['emp_length'] = df['emp_length'].map(emp_map)

        # 3. Clean Dates & Calculate Credit History Age
        # Parse issue date
        df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y', errors='coerce')
        # Parse earliest credit line
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y', errors='coerce')
        
        # Adjust earliest_cr_line for years parsed in the future (e.g. 2068 instead of 1968)
        # Lending Club data contains older history, so if the parsed year is > issue year, subtract 100 years.
        def adjust_century(row):
            if pd.isnull(row['earliest_cr_line']) or pd.isnull(row['issue_d']):
                return row['earliest_cr_line']
            if row['earliest_cr_line'].year > row['issue_d'].year:
                try:
                    return row['earliest_cr_line'].replace(year=row['earliest_cr_line'].year - 100)
                except ValueError: # handle leap years/bounds if needed
                    return row['earliest_cr_line'] - pd.DateOffset(years=100)
            return row['earliest_cr_line']
            
        df['earliest_cr_line'] = df.apply(adjust_century, axis=1)
        
        # Calculate credit history age in months at time of loan issue
        df['credit_history_age'] = ((df['issue_d'] - df['earliest_cr_line']).dt.days / 30.4375).round()
        # Cap age at 0 in case of minor date anomalies
        df['credit_history_age'] = df['credit_history_age'].apply(lambda x: max(0, x) if not pd.isnull(x) else x)

        # 4. Clean up Target Variable (loan_status)
        # Exclude 'Current' loans as their final default status is undetermined
        df = df[df['loan_status'] != 'Current'].copy()
        
        # Define target variable: Default = 1, Fully Paid = 0
        df['target'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
        
        # Keep track of the issue year/month for out-of-time splits
        df['issue_year'] = df['issue_d'].dt.year
        df['issue_month'] = df['issue_d'].dt.month

        # Save cleaned dataframe
        self.cleaned_df = df
        return df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into In-Time (2007-2010) and Out-of-Time (2011) populations.
        """
        if self.cleaned_df is None:
            self.clean_data()
            
        # OOT is loans issued in 2011. In-Time is loans issued before 2011 (2007-2010).
        in_time_df = self.cleaned_df[self.cleaned_df['issue_year'] < 2011].copy()
        oot_df = self.cleaned_df[self.cleaned_df['issue_year'] == 2011].copy()
        
        return in_time_df, oot_df
