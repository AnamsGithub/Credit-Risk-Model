import json
import os

def create_notebook(filename: str, cells: list):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    path = os.path.join("notebooks", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {path}")

def make_cell(cell_type: str, source: list):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source]
    }

def main():
    # ==========================================
    # 01_EDA.ipynb
    # ==========================================
    eda_cells = [
        make_cell("markdown", [
            "# Phase 1: Exploratory Data Analysis & Data Quality Assessment",
            "This notebook profiles the Lending Club loan dataset to understand key borrower risk characteristics, assess data quality (missing values, duplicates, outliers), and calculate the baseline portfolio default rate."
        ]),
        make_cell("code", [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "import os",
            "import sys",
            "",
            "# Ensure we are running from the project root directory",
            "if os.path.basename(os.getcwd()) == 'notebooks':",
            "    os.chdir('..')",
            "",
            "sys.path.append(os.path.abspath('src'))",
            "from data_processing import DataProcessor"
        ]),
        make_cell("markdown", [
            "## 1. Load Data",
            "We load the Lending Club loan dataset and inspect its dimensions."
        ]),
        make_cell("code", [
            "processor = DataProcessor('data/loan.csv')",
            "df = processor.load_data()",
            "print('Dataset Dimensions:', df.shape)"
        ]),
        make_cell("markdown", [
            "## 2. Data Quality Assessment",
            "We check for missing values, duplicates, and column types."
        ]),
        make_cell("code", [
            "print('Duplicate rows:', df.duplicated().sum())",
            "missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)",
            "print('\\nTop Columns with Missing Values (%):\\n', missing_pct.head(15))"
        ]),
        make_cell("markdown", [
            "## 3. Data Cleaning & Feature Calculations",
            "Clean formatted percentage columns, convert employment length into numeric values, and parse credit dates to calculate credit history age in months."
        ]),
        make_cell("code", [
            "df_clean = processor.clean_data()",
            "print('Resolved dataset shape (excluding Current loans):', df_clean.shape)",
            "print('Overall default rate:', round(df_clean['target'].mean() * 100, 2), '%')"
        ]),
        make_cell("markdown", [
            "## 4. Visualizing Risk Characteristics",
            "Let's look at the distribution of borrower income, loan sizes, interest rates, and debt burdens."
        ]),
        make_cell("code", [
            "sns.set_theme(style='whitegrid')",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "",
            "# Loan Amount Distribution",
            "sns.histplot(df_clean['loan_amnt'], kde=True, ax=axes[0,0], color='blue')",
            "axes[0,0].set_title('Loan Amount Distribution')",
            "",
            "# Interest Rate Distribution",
            "sns.histplot(df_clean['int_rate'], kde=True, ax=axes[0,1], color='orange')",
            "axes[0,1].set_title('Interest Rate Distribution')",
            "",
            "# Annual Income Distribution (capped for display)",
            "sns.histplot(df_clean[df_clean['annual_inc'] < 150000]['annual_inc'], kde=True, ax=axes[1,0], color='green')",
            "axes[1,0].set_title('Annual Income Distribution (< $150k)')",
            "",
            "# DTI Ratio Distribution",
            "sns.histplot(df_clean['dti'], kde=True, ax=axes[1,1], color='purple')",
            "axes[1,1].set_title('Debt-to-Income (DTI) Ratio')",
            "",
            "plt.tight_layout()",
            "plt.show()"
        ]),
        make_cell("markdown", [
            "## 5. Category Delinquency Analysis",
            "Analyze the default rate across categories such as Home Ownership and Loan Purpose."
        ]),
        make_cell("code", [
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))",
            "",
            "# Default rate by Home Ownership",
            "sns.barplot(x='home_ownership', y='target', data=df_clean, ax=axes[0], palette='viridis', errorbar=None)",
            "axes[0].set_title('Default Rate by Home Ownership')",
            "axes[0].set_ylabel('Default Rate')",
            "",
            "# Default rate by Loan Purpose",
            "sns.barplot(x='target', y='purpose', data=df_clean, ax=axes[1], palette='rocket', errorbar=None)",
            "axes[1].set_title('Default Rate by Loan Purpose')",
            "axes[1].set_xlabel('Default Rate')",
            "",
            "plt.tight_layout()",
            "plt.show()"
        ])
    ]
    create_notebook("01_EDA.ipynb", eda_cells)

    # ==========================================
    # 02_WOE_Binning.ipynb
    # ==========================================
    bin_cells = [
        make_cell("markdown", [
            "# Phase 2: Weight of Evidence (WoE) Binning & Information Value (IV) Feature Selection",
            "This notebook implements true credit scorecard feature engineering. We bin numerical and categorical variables, replace raw inputs with their WoE values, and calculate Information Value (IV) to select stable risk features."
        ]),
        make_cell("code", [
            "import pandas as pd",
            "import os",
            "import sys",
            "",
            "# Ensure we are running from the project root directory",
            "if os.path.basename(os.getcwd()) == 'notebooks':",
            "    os.chdir('..')",
            "",
            "sys.path.append(os.path.abspath('src'))",
            "from data_processing import DataProcessor",
            "from woe_binning import WoEBinning"
        ]),
        make_cell("markdown", [
            "## 1. Load Cleaned Splits",
            "We load the cleaned splits (Train/Test and Out-of-Time)."
        ]),
        make_cell("code", [
            "processor = DataProcessor('data/loan.csv')",
            "processor.clean_data()",
            "train_df, oot_df = processor.split_data()",
            "print('Train size:', train_df.shape)",
            "print('OOT size:', oot_df.shape)"
        ]),
        make_cell("markdown", [
            "## 2. Fit WoE Mappings",
            "We fit automated binning (Decision-Tree based for numeric and class groupings for categorical) on the training set."
        ]),
        make_cell("code", [
            "numeric_features = [",
            "    'loan_amnt', 'annual_inc', 'dti', 'revol_util', ",
            "    'delinq_2yrs', 'inq_last_6mths', 'open_acc', ",
            "    'pub_rec', 'pub_rec_bankruptcies', 'credit_history_age',",
            "    'emp_length'",
            "]",
            "categorical_features = ['home_ownership', 'purpose']",
            "",
            "woe_model = WoEBinning(target_col='target')",
            "woe_model.fit_all(train_df, numeric_features, categorical_features)"
        ]),
        make_cell("markdown", [
            "## 3. Information Value (IV) Ranking & Selection",
            "Rank characteristics based on their predictive power (IV) and filter out useless (<0.02) and suspicious (>0.50) features."
        ]),
        make_cell("code", [
            "iv_df = pd.read_csv('outputs/scorecards/iv_report.csv') if pd.io.common.file_exists('outputs/scorecards/iv_report.csv') else pd.DataFrame(woe_model.iv_report)",
            "print('IV Feature Rankings:')",
            "print(iv_df)"
        ]),
        make_cell("markdown", [
            "## 4. Inspecting WoE Lookup Tables",
            "Let's look at the generated Weight of Evidence mapping rules for a key feature like `annual_inc`."
        ]),
        make_cell("code", [
            "woe_df = pd.read_csv('outputs/scorecards/woe_tables.csv') if pd.io.common.file_exists('outputs/scorecards/woe_tables.csv') else None",
            "if woe_df is not None:",
            "    print(woe_df[woe_df['variable'] == 'annual_inc'])"
        ]),
        make_cell("markdown", [
            "## 5. WoE Transformation",
            "We transform the raw training features to WoE values for model input."
        ]),
        make_cell("code", [
            "train_woe = woe_model.transform(train_df)",
            "print('Transformed WoE DataFrame Head:')",
            "print(train_woe.head())"
        ])
    ]
    create_notebook("02_WOE_Binning.ipynb", bin_cells)

    # ==========================================
    # 03_Logistic_Scorecard.ipynb
    # ==========================================
    scorecard_cells = [
        make_cell("markdown", [
            "# Phase 3: Logistic Regression Scorecard Model",
            "In this notebook, we build the champion credit scorecard. We train a Logistic Regression model on the selected WoE features, check model coefficients and multicollinearity (VIF), and scale log-odds into score points."
        ]),
        make_cell("code", [
            "import pandas as pd",
            "import os",
            "import sys",
            "import yaml",
            "",
            "# Ensure we are running from the project root directory",
            "if os.path.basename(os.getcwd()) == 'notebooks':",
            "    os.chdir('..')",
            "",
            "sys.path.append(os.path.abspath('src'))",
            "from data_processing import DataProcessor",
            "from woe_binning import WoEBinning",
            "from scorecard import ScorecardModel"
        ]),
        make_cell("markdown", [
            "## 1. Load Data & Mappings",
            "Load the dataset splits and the WoE mapping object."
        ]),
        make_cell("code", [
            "processor = DataProcessor('data/loan.csv')",
            "processor.clean_data()",
            "train_df, oot_df = processor.split_data()",
            "",
            "woe_model = WoEBinning(target_col='target')",
            "woe_model.fit_all(train_df, [",
            "    'loan_amnt', 'annual_inc', 'dti', 'revol_util', ",
            "    'delinq_2yrs', 'inq_last_6mths', 'open_acc', ",
            "    'pub_rec', 'pub_rec_bankruptcies', 'credit_history_age',",
            "    'emp_length'",
            "], ['home_ownership', 'purpose'])",
            "",
            "train_woe = woe_model.transform(train_df)",
            "selected_woe_cols = [f'{col}_woe' for col in woe_model.selected_features]"
        ]),
        make_cell("markdown", [
            "## 2. Train Logistic Regression",
            "We train the regression model predicting default (target=1) using statsmodels."
        ]),
        make_cell("code", [
            "scorecard = ScorecardModel(base_score=600, base_odds=50, pdo=20)",
            "scorecard.fit(train_woe[selected_woe_cols], train_woe['target'])",
            "print(scorecard.model.summary())"
        ]),
        make_cell("markdown", [
            "## 3. Multicollinearity & Diagnostic Checks",
            "We check Variance Inflation Factors (VIF) and verify that all coefficients are statistically significant (p < 0.05) and have the expected sign (negative for default prediction)."
        ]),
        make_cell("code", [
            "print('Variance Inflation Factor (VIF) Results:')",
            "print(scorecard.vif_report)",
            "",
            "is_valid, issues = scorecard.validate_coefficients()",
            "print('\\nModel Diagnostics Check:')",
            "if is_valid:",
            "    print('SUCCESS: All coefficient signs are correct and VIF values are within regulatory limits.')",
            "else:",
            "    print('WARNING Issues Detected:')",
            "    for issue in issues:",
            "        print(' -', issue)"
        ]),
        make_cell("markdown", [
            "## 4. Scorecard Scaling & Points Mapping",
            "Translate the logistic regression coefficients into integer-based score contributions per bin."
        ]),
        make_cell("code", [
            "scorecard.build_scorecard_table(woe_model.mappings)",
            "scorecard_table = scorecard.scorecard_table",
            "print('Scorecard Points Table (First 15 Rows):')",
            "print(scorecard_table.head(15))"
        ]),
        make_cell("markdown", [
            "## 5. Customer Score Generation",
            "Demonstrate credit scoring on train sample and show the output scores and PD values."
        ]),
        make_cell("code", [
            "scores = scorecard.predict_score(train_woe)",
            "pds = scorecard.predict_pd(train_woe)",
            "scored_df = pd.DataFrame({'Target': train_woe['target'], 'Credit Score': scores, 'PD': pds})",
            "print(scored_df.describe())"
        ])
    ]
    create_notebook("03_Logistic_Scorecard.ipynb", scorecard_cells)

    # ==========================================
    # 04_XGBoost_Challenger.ipynb
    # ==========================================
    xgb_cells = [
        make_cell("markdown", [
            "# Phase 4: XGBoost Challenger Model & SHAP analysis",
            "This notebook trains an XGBoost classifier as a challenger model. We tune hyperparameters using grid search, compare its raw performance against the scorecard, and execute SHAP analysis to extract risk drivers."
        ]),
        make_cell("code", [
            "import pandas as pd",
            "import os",
            "import sys",
            "",
            "# Ensure we are running from the project root directory",
            "if os.path.basename(os.getcwd()) == 'notebooks':",
            "    os.chdir('..')",
            "",
            "sys.path.append(os.path.abspath('src'))",
            "from data_processing import DataProcessor",
            "from woe_binning import WoEBinning",
            "from xgboost_model import XGBoostChallenger"
        ]),
        make_cell("markdown", [
            "## 1. Load Data Splits",
            "Load raw splits since trees handle missing values directly."
        ]),
        make_cell("code", [
            "processor = DataProcessor('data/loan.csv')",
            "processor.clean_data()",
            "train_df, oot_df = processor.split_data()",
            "",
            "woe_model = WoEBinning(target_col='target')",
            "woe_model.fit_all(train_df, [",
            "    'loan_amnt', 'annual_inc', 'dti', 'revol_util', ",
            "    'delinq_2yrs', 'inq_last_6mths', 'open_acc', ",
            "    'pub_rec', 'pub_rec_bankruptcies', 'credit_history_age',",
            "    'emp_length'",
            "], ['home_ownership', 'purpose'])"
        ]),
        make_cell("markdown", [
            "## 2. Train XGBoost Model",
            "Configure categorical label encodings and fit the tuned tree models."
        ]),
        make_cell("code", [
            "xgb_cols = woe_model.selected_features",
            "X_train = train_df[xgb_cols].copy()",
            "X_oot = oot_df[xgb_cols].copy()",
            "",
            "# Ordinal encoding for categories",
            "for col in ['home_ownership', 'purpose']:",
            "    if col in xgb_cols:",
            "        X_train[col] = X_train[col].astype('category').cat.codes",
            "        X_oot[col] = X_oot[col].astype('category').cat.codes",
            "",
            "challenger = XGBoostChallenger(target_col='target')",
            "challenger.fit_and_tune(X_train, train_df['target'], X_oot, oot_df['target'])"
        ]),
        make_cell("markdown", [
            "## 3. SHAP Analysis",
            "Generate local and global SHAP summary explanations to identify top risk drivers."
        ]),
        make_cell("code", [
            "import matplotlib.pyplot as plt",
            "# Generate shap files",
            "challenger.run_shap_analysis(X_train, 'outputs/reports/shap_analysis.html', 'outputs/reports/shap_summary.png')",
            "print('SHAP charts saved. High-risk drivers can be inspected in output files.')"
        ]),
        make_cell("markdown", [
            "## 4. Feature Importance Rankings",
            "Display standard XGBoost feature gain importances."
        ]),
        make_cell("code", [
            "imp_df = challenger.get_feature_importance()",
            "print(imp_df)"
        ])
    ]
    create_notebook("04_XGBoost_Challenger.ipynb", xgb_cells)

    # ==========================================
    # 05_Model_Validation.ipynb
    # ==========================================
    val_cells = [
        make_cell("markdown", [
            "# Phase 5: Model Validation & Monitoring",
            "This notebook executes banking-grade model validation. We compute ROC-AUC, KS, Gini, and Brier scores, plot calibration curves, and evaluate temporal stability using Population Stability Index (PSI) and Characteristic Stability Index (CSI) against the Out-of-Time validation cohort."
        ]),
        make_cell("code", [
            "import pandas as pd",
            "import os",
            "import sys",
            "",
            "# Ensure we are running from the project root directory",
            "if os.path.basename(os.getcwd()) == 'notebooks':",
            "    os.chdir('..')",
            "",
            "sys.path.append(os.path.abspath('src'))",
            "from data_processing import DataProcessor",
            "from woe_binning import WoEBinning",
            "from scorecard import ScorecardModel",
            "from validation import ModelValidator",
            "from stability import StabilityMonitor"
        ]),
        make_cell("markdown", [
            "## 1. Load Model Predictions",
            "We load scored datasets to run performance checks."
        ]),
        make_cell("code", [
            "processor = DataProcessor('data/loan.csv')",
            "processor.clean_data()",
            "train_df, oot_df = processor.split_data()",
            "",
            "woe_model = WoEBinning(target_col='target')",
            "woe_model.fit_all(train_df, [",
            "    'loan_amnt', 'annual_inc', 'dti', 'revol_util', ",
            "    'delinq_2yrs', 'inq_last_6mths', 'open_acc', ",
            "    'pub_rec', 'pub_rec_bankruptcies', 'credit_history_age',",
            "    'emp_length'",
            "], ['home_ownership', 'purpose'])",
            "",
            "train_woe = woe_model.transform(train_df)",
            "oot_woe = woe_model.transform(oot_df)",
            "selected_woe_cols = [f'{col}_woe' for col in woe_model.selected_features]",
            "",
            "scorecard = ScorecardModel()",
            "scorecard.fit(train_woe[selected_woe_cols], train_woe['target'])",
            "scorecard.build_scorecard_table(woe_model.mappings)",
            "",
            "train_scores = scorecard.predict_score(train_woe)",
            "oot_scores = scorecard.predict_score(oot_woe)",
            "oot_pds = scorecard.predict_pd(oot_woe)"
        ]),
        make_cell("markdown", [
            "## 2. Credit Performance Curves",
            "Plot ROC and KS curves to evaluate risk differentiation."
        ]),
        make_cell("code", [
            "y_true_oot = oot_df['target'].values",
            "ModelValidator.plot_roc_curve(y_true_oot, oot_pds, 'ROC Curve - Out-of-Time Validation')",
            "ModelValidator.plot_ks_curve(y_true_oot, oot_pds, 'KS Separation Curve - Out-of-Time Validation')"
        ]),
        make_cell("markdown", [
            "## 3. Decile Analysis",
            "Verify that default rates decrease monotonically as credit scores increase (which corresponds to higher deciles of score or lower deciles of default probability)."
        ]),
        make_cell("code", [
            "decile_df = ModelValidator.generate_decile_analysis(y_true_oot, oot_pds)",
            "print(decile_df[['decile', 'total_loans', 'defaults', 'default_rate', 'ks_stat', 'lift']])"
        ]),
        make_cell("markdown", [
            "## 4. Temporal Stability & Drift: PSI & CSI",
            "Calculate Population Stability Index (PSI) and Characteristic Stability Index (CSI) between In-Time training and Out-of-Time validation cohorts."
        ]),
        make_cell("code", [
            "psi_val, psi_details = StabilityMonitor.calculate_psi(train_scores, oot_scores)",
            "print(f'Total Portfolio Credit Score PSI: {psi_val:.4f}')",
            "if psi_val < 0.10:",
            "    print('Status: STABLE (No action required)')",
            "elif psi_val < 0.25:",
            "    print('Status: MODERATE SHIFT (Monitor closely)')",
            "else:",
            "    print('Status: SIGNIFICANT DRIFT (Requires retraining)')",
            "",
            "csi_vals, _ = StabilityMonitor.calculate_csi(train_df, oot_df, woe_model.mappings)",
            "print('\\nTop CSI Variables (Characteristic Drift):')",
            "for col, val in sorted(csi_vals.items(), key=lambda x: x[1], reverse=True)[:5]:",
            "    print(f' - {col}: {val:.4f}')"
        ])
    ]
    create_notebook("05_Model_Validation.ipynb", val_cells)

    # ==========================================
    # 06_Policy_Simulation.ipynb
    # ==========================================
    policy_cells = [
        make_cell("markdown", [
            "# Phase 6: Business Policy Simulation",
            "This notebook translates scorecard mathematical probability models into risk-adjusted credit approval decisions. We establish credit risk bands, run credit policy cutoff simulations, and estimate portfolio expected revenue, expected losses, and profit margins."
        ]),
        make_cell("code", [
            "import pandas as pd",
            "import os",
            "import sys",
            "",
            "# Ensure we are running from the project root directory",
            "if os.path.basename(os.getcwd()) == 'notebooks':",
            "    os.chdir('..')",
            "",
            "sys.path.append(os.path.abspath('src'))",
            "from data_processing import DataProcessor",
            "from woe_binning import WoEBinning",
            "from scorecard import ScorecardModel",
            "from business_decision_engine import BusinessDecisionEngine",
            "from policy_sim import PolicySimulatorPlotter"
        ]),
        make_cell("markdown", [
            "## 1. Load Scored Portfolio",
            "We load scored OOT data for credit decision analysis."
        ]),
        make_cell("code", [
            "processor = DataProcessor('data/loan.csv')",
            "processor.clean_data()",
            "train_df, oot_df = processor.split_data()",
            "",
            "woe_model = WoEBinning(target_col='target')",
            "woe_model.fit_all(train_df, [",
            "    'loan_amnt', 'annual_inc', 'dti', 'revol_util', ",
            "    'delinq_2yrs', 'inq_last_6mths', 'open_acc', ",
            "    'pub_rec', 'pub_rec_bankruptcies', 'credit_history_age',",
            "    'emp_length'",
            "], ['home_ownership', 'purpose'])",
            "",
            "train_woe = woe_model.transform(train_df)",
            "oot_woe = woe_model.transform(oot_df)",
            "selected_woe_cols = [f'{col}_woe' for col in woe_model.selected_features]",
            "",
            "scorecard = ScorecardModel()",
            "scorecard.fit(train_woe[selected_woe_cols], train_woe['target'])",
            "scorecard.build_scorecard_table(woe_model.mappings)",
            "",
            "oot_scores = scorecard.predict_score(oot_woe)",
            "oot_pds = scorecard.predict_pd(oot_woe)",
            "",
            "df_scores_pd = oot_df[['loan_amnt', 'target']].copy()",
            "df_scores_pd['score'] = oot_scores",
            "df_scores_pd['pd'] = oot_pds"
        ]),
        make_cell("markdown", [
            "## 2. Portfolio Risk Bands",
            "Break down the credit portfolio by risk categories: Excellent (750+), Good (700-749), Moderate (650-699), High Risk (600-649), and Very High Risk (<600)."
        ]),
        make_cell("code", [
            "decision_engine = BusinessDecisionEngine(avg_interest_rate=0.12, loss_given_default=0.60, avg_loan_term_years=3.0)",
            "band_summary = decision_engine.summarize_risk_bands(df_scores_pd)",
            "print(band_summary[['risk_band', 'total_loans', 'total_amount', 'mean_pd', 'expected_revenue', 'expected_loss', 'net_profit']])"
        ]),
        make_cell("markdown", [
            "## 3. Lending Cutoff Policy Simulation",
            "Simulate portfolio key performance indicators (KPIs) at cutoff scores ranging from 500 to 800."
        ]),
        make_cell("code", [
            "cutoff_sim = decision_engine.run_cutoff_simulation(df_scores_pd)",
            "print('Simulation results for selected cutoffs:')",
            "print(cutoff_sim[cutoff_sim['cutoff'].isin([580, 600, 620, 640, 660, 700])])"
        ]),
        make_cell("markdown", [
            "## 4. Policy Trade-off Curves",
            "Plot Approval Rate vs Expected Portfolio Default Rate and projected Net Profit curves."
        ]),
        make_cell("code", [
            "PolicySimulatorPlotter.plot_tradeoff_curves(cutoff_sim)",
            "PolicySimulatorPlotter.plot_profit_optimization(cutoff_sim)"
        ]),
        make_cell("markdown", [
            "## 5. Executive Lending Guidance",
            "Recommend optimal decision thresholds for management review."
        ]),
        make_cell("code", [
            "rec = decision_engine.recommend_optimal_cutoff(cutoff_sim)",
            "print('Optimal Lending Strategies:')",
            "print(f' - Profit-Maximizing: Cutoff {rec[\"profit_max\"][\"cutoff\"]}, Profit: ${rec[\"profit_max\"][\"net_profit\"]/1e6:.2f}M, Approval: {rec[\"profit_max\"][\"approval_rate\"]:.1f}%')",
            "print(f' - Balanced (Low Bad Rate): Cutoff {rec[\"balanced\"][\"cutoff\"]}, Profit: ${rec[\"balanced\"][\"net_profit\"]/1e6:.2f}M, Approval: {rec[\"balanced\"][\"approval_rate\"]:.1f}%')",
            "print(f' - Conservative: Cutoff {rec[\"conservative\"][\"cutoff\"]}, Profit: ${rec[\"conservative\"][\"net_profit\"]/1e6:.2f}M, Approval: {rec[\"conservative\"][\"approval_rate\"]:.1f}%')"
        ])
    ]
    create_notebook("06_Policy_Simulation.ipynb", policy_cells)

if __name__ == "__main__":
    main()
