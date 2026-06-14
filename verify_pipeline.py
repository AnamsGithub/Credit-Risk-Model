import os
import sys
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add src to python path
sys.path.append(os.path.abspath('src'))

# pyrefly: ignore [missing-import]
from data_processing import DataProcessor
# pyrefly: ignore [missing-import]
from woe_binning import WoEBinning
# pyrefly: ignore [missing-import]
from scorecard import ScorecardModel
# pyrefly: ignore [missing-import]
from xgboost_model import XGBoostChallenger
# pyrefly: ignore [missing-import]
from validation import ModelValidator
# pyrefly: ignore [missing-import]
from stability import StabilityMonitor
# pyrefly: ignore [missing-import]
from score_monitoring import ScoreMonitor
# pyrefly: ignore [missing-import]
from business_decision_engine import BusinessDecisionEngine
# pyrefly: ignore [missing-import]
from policy_sim import PolicySimulatorPlotter
# pyrefly: ignore [missing-import]
from pdf_generator import PDFReportGenerator
# pyrefly: ignore [missing-import]
from ppt_generator import PPTPresentationGenerator
from lgd_model import LGDModel
from ead_model import EADModel

def main():
    print("==================================================")
    # 0. Load scaling configurations
    print("0. Loading config.yaml...")
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    scaling_cfg = config['scaling']
    fin_cfg = config['financials']
    
    # Create output directories if they don't exist
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/scorecards", exist_ok=True)
    os.makedirs("outputs/monitoring", exist_ok=True)

    # 1. Data Processing
    print("\n1. Running Data Cleaning & Splits...")
    processor = DataProcessor("data/loan.csv")
    processor.clean_data()
    train_df, oot_df = processor.split_data()
    print(f"Train/Test (In-Time) shape: {train_df.shape}")
    print(f"Out-of-Time (OOT) shape: {oot_df.shape}")
    
    # 2. WoE Binning & Feature Selection
    print("\n2. Fitting Weight of Evidence (WoE) Mappings...")
    # Select predictors known at application time
    numeric_features = [
        'loan_amnt', 'annual_inc', 'dti', 'revol_util', 
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 
        'pub_rec', 'pub_rec_bankruptcies', 'credit_history_age',
        'emp_length'
    ]
    categorical_features = ['home_ownership', 'purpose']
    
    woe_model = WoEBinning(target_col='target')
    woe_model.fit_all(train_df, numeric_features, categorical_features)
    
    # Save mappings
    woe_model.save_mappings("outputs/scorecards/woe_mapping.json")
    woe_model.export_reports(
        "outputs/scorecards/woe_tables.csv",
        "outputs/scorecards/iv_report.csv",
        "outputs/scorecards/feature_dictionary.xlsx"
    )
    print("Features selected by IV (0.02 - 0.50):", woe_model.selected_features)
    
    # Transform data
    train_woe = woe_model.transform(train_df)
    oot_woe = woe_model.transform(oot_df)
    
    # 3. Fit Champion Scorecard Model
    print("\n3. Training Logistic Regression Champion Model...")
    woe_feature_cols = [f"{col}_woe" for col in woe_model.selected_features]
    X_train = train_woe[woe_feature_cols]
    y_train = train_woe['target']
    
    # Train scorecard
    scorecard = ScorecardModel(
        base_score=scaling_cfg['base_score'],
        base_odds=scaling_cfg['base_odds'],
        pdo=scaling_cfg['pdo'],
        min_score=scaling_cfg['minimum_score'],
        max_score=scaling_cfg['maximum_score']
    )
    scorecard.fit(X_train, y_train)
    
    # Check VIF and validation issues
    vif_report = scorecard.vif_report
    is_valid, coef_issues = scorecard.validate_coefficients()
    print("VIF Diagnostics:\n", vif_report)
    if not is_valid:
        print("WARNING scorecard diagnostics issues found:")
        for issue in coef_issues:
            print(f"  - {issue}")
            
    # Build Scorecard Lookup Table
    scorecard.build_scorecard_table(woe_model.mappings)
    scorecard.save_scorecard("outputs/scorecards/scorecard_table.csv")
    
    # Score the populations
    train_scores = scorecard.predict_score(train_woe)
    oot_scores = scorecard.predict_score(oot_woe)
    
    train_pds = scorecard.predict_pd(train_woe)
    oot_pds = scorecard.predict_pd(oot_woe)
    
    # 4. Train Challenger XGBoost Model
    print("\n4. Training XGBoost Challenger Model...")
    # Use raw variables for XGBoost (handle missing by leaving as NaN, tree deals with it)
    xgb_train_cols = woe_model.selected_features
    # Simple label encoding for categorical variables inside train_df
    X_train_raw = train_df[xgb_train_cols].copy()
    X_oot_raw = oot_df[xgb_train_cols].copy()
    
    for col in categorical_features:
        if col in xgb_train_cols:
            # Code categories as ordinal integers for XGBoost
            X_train_raw[col] = X_train_raw[col].astype('category').cat.codes
            X_oot_raw[col] = X_oot_raw[col].astype('category').cat.codes
            
    # Train XGBoost
    xgb_challenger = XGBoostChallenger(target_col='target')
    # Use validation set
    xgb_challenger.fit_and_tune(X_train_raw, y_train, X_oot_raw, oot_df['target'])
    
    # SHAP Plot paths
    shap_html = "outputs/reports/shap_analysis.html"
    shap_png = "outputs/reports/shap_summary.png"
    xgb_challenger.run_shap_analysis(X_train_raw, shap_html, shap_png)
    
    # Challenger PD predictions
    train_pds_xgb = xgb_challenger.predict_pd(X_train_raw)
    oot_pds_xgb = xgb_challenger.predict_pd(X_oot_raw)
    
    # 5. Model Validation & Performance Metrics
    print("\n5. Running Model Validation...")
    # Champion metrics (OOT)
    y_true_oot = oot_df['target'].values
    champ_auc = roc_auc_score(y_true_oot, oot_pds)
    champ_gini = 2 * champ_auc - 1
    champ_ks, champ_ks_thresh, _, _, _ = ModelValidator.calculate_ks(y_true_oot, oot_pds)
    champ_brier = ModelValidator.brier_score(y_true_oot, oot_pds)
    
    champ_pred_labels = (oot_pds > 0.5).astype(int)
    champ_prec = precision_score(y_true_oot, champ_pred_labels, zero_division=0)
    champ_recall = recall_score(y_true_oot, champ_pred_labels, zero_division=0)
    champ_f1 = f1_score(y_true_oot, champ_pred_labels, zero_division=0)
    
    champion_metrics = {
        'auc': champ_auc, 'gini': champ_gini, 'ks': champ_ks, 
        'brier': champ_brier, 'precision': champ_prec, 'recall': champ_recall, 'f1': champ_f1
    }
    
    # Challenger metrics (OOT)
    chall_auc = roc_auc_score(y_true_oot, oot_pds_xgb)
    chall_gini = 2 * chall_auc - 1
    chall_ks, _, _, _, _ = ModelValidator.calculate_ks(y_true_oot, oot_pds_xgb)
    chall_brier = ModelValidator.brier_score(y_true_oot, oot_pds_xgb)
    
    chall_pred_labels = (oot_pds_xgb > 0.5).astype(int)
    chall_prec = precision_score(y_true_oot, chall_pred_labels, zero_division=0)
    chall_recall = recall_score(y_true_oot, chall_pred_labels, zero_division=0)
    chall_f1 = f1_score(y_true_oot, chall_pred_labels, zero_division=0)
    
    challenger_metrics = {
        'auc': chall_auc, 'gini': chall_gini, 'ks': chall_ks, 
        'brier': chall_brier, 'precision': chall_prec, 'recall': chall_recall, 'f1': chall_f1
    }
    
    # Save performance curves
    roc_img = "outputs/reports/roc_curve.png"
    ks_img = "outputs/reports/ks_curve.png"
    cal_img = "outputs/reports/calibration_curve.png"
    
    ModelValidator.plot_roc_curve(y_true_oot, oot_pds, "ROC Curve (Champion Scorecard OOT)", roc_img)
    ModelValidator.plot_ks_curve(y_true_oot, oot_pds, "KS Curve (Champion Scorecard OOT)", ks_img)
    ModelValidator.plot_calibration_curve(y_true_oot, oot_pds, "Calibration Curve (Champion Scorecard OOT)", cal_img)
    
    # Generate Decile Analysis
    decile_df = ModelValidator.generate_decile_analysis(y_true_oot, oot_pds)
    decile_df.to_csv("outputs/reports/decile_analysis.csv", index=False)
    
    # Export Champion Challenger Report Excel
    champ_chall_data = [
        {"Model": "Champion Logistic Scorecard", "ROC-AUC": champ_auc, "KS": champ_ks, "Gini": champ_gini, "Brier Score": champ_brier, "Precision": champ_prec, "Recall": champ_recall, "F1-Score": champ_f1},
        {"Model": "Challenger XGBoost", "ROC-AUC": chall_auc, "KS": chall_ks, "Gini": chall_gini, "Brier Score": chall_brier, "Precision": chall_prec, "Recall": chall_recall, "F1-Score": chall_f1}
    ]
    pd.DataFrame(champ_chall_data).to_excel("outputs/monitoring/champion_challenger_report.xlsx", index=False)
    print("Champion vs Challenger report saved.")
    
    # 6. Stability Analysis (PSI & CSI)
    print("\n6. Running Stability Analyses (PSI & CSI)...")
    psi_val, psi_df = StabilityMonitor.calculate_psi(train_scores, oot_scores)
    csi_vals, csi_details = StabilityMonitor.calculate_csi(train_df, oot_df, woe_model.mappings)
    
    StabilityMonitor.export_reports(
        psi_val, psi_df, csi_vals, csi_details,
        "outputs/monitoring/psi_report.xlsx",
        "outputs/monitoring/csi_report.xlsx"
    )
    print(f"Total Credit Score PSI: {psi_val:.4f}")
    
    # 7. Score Monitoring
    print("\n7. Running Score Monitoring & Migration Matrix...")
    monitor = ScoreMonitor(
        min_score=scaling_cfg['minimum_score'],
        max_score=scaling_cfg['maximum_score']
    )
    score_stats = monitor.analyze_score_distribution(train_scores, oot_scores)
    migration_matrix = monitor.create_migration_matrix(oot_df, oot_scores)
    
    dist_img = "outputs/reports/score_distribution.png"
    band_img = "outputs/reports/risk_band_distribution.png"
    monitor.plot_score_comparison(train_scores, oot_scores, dist_img)
    monitor.plot_risk_band_comparison(train_scores, oot_scores, band_img)
    
    monitor.export_reports(score_stats, migration_matrix, "outputs/monitoring/score_monitoring_report.xlsx")
    
    # 8. Business Policy Simulation
    print("\n8. Running Business Policy & Lending Simulations...")
    decision_engine = BusinessDecisionEngine(
        avg_interest_rate=fin_cfg['avg_interest_rate'],
        loss_given_default=fin_cfg['loss_given_default'],
        avg_loan_term_years=fin_cfg['avg_loan_term_years']
    )
    
    df_scores_pd = oot_df[['loan_amnt', 'target']].copy()
    df_scores_pd['score'] = oot_scores
    df_scores_pd['pd'] = oot_pds
    
    risk_band_summary = decision_engine.summarize_risk_bands(df_scores_pd)
    risk_band_summary.to_csv("outputs/reports/risk_band_summary.csv", index=False)
    
    cutoff_sim = decision_engine.run_cutoff_simulation(df_scores_pd)
    cutoff_sim.to_csv("outputs/reports/approval_rate_analysis.csv", index=False)
    
    recommend = decision_engine.recommend_optimal_cutoff(cutoff_sim)
    
    # Plot tradeoff curves
    tradeoff_img = "outputs/reports/policy_tradeoff_curve.png"
    profit_img = "outputs/reports/profit_optimization_curve.png"
    PolicySimulatorPlotter.plot_tradeoff_curves(cutoff_sim, tradeoff_img)
    PolicySimulatorPlotter.plot_profit_optimization(cutoff_sim, profit_img)
    
    # 8.5. Loss Given Default (LGD) and Exposure at Default (EAD) Modeling
    print("\n8.5. Training LGD and EAD Models...")
    lgd_model = LGDModel()
    metrics_lgd = lgd_model.fit(train_df, oot_df)
    lgd_model.save_model("outputs/scorecards/lgd_model.pkl")
    
    # Predict and save LGD predictions for defaults
    oot_defaults = oot_df[oot_df['target'] == 1].copy()
    if len(oot_defaults) == 0 and 'loan_status' in oot_df.columns:
        oot_defaults = oot_df[oot_df['loan_status'].isin(['Charged Off', 'Default'])].copy()
        
    oot_defaults['actual_lgd'] = lgd_model.calculate_lgd_target(oot_defaults)
    oot_defaults['pred_lgd'] = lgd_model.predict_lgd(oot_defaults)
    oot_defaults[['id', 'member_id', 'actual_lgd', 'pred_lgd']].to_csv("outputs/scorecards/lgd_predictions.csv", index=False)
    lgd_model.generate_report(train_df, oot_df, "outputs/reports/lgd_model_report.pdf", metrics_lgd)
    
    ead_model = EADModel()
    metrics_ead = ead_model.fit(train_df, oot_df)
    ead_model.save_model("outputs/scorecards/ead_model.pkl")
    
    # Predict and save EAD predictions for defaults
    oot_defaults['actual_ead_pct'] = ead_model.calculate_ead_target(oot_defaults)
    oot_defaults['pred_ead_pct'] = ead_model.predict_ead(oot_defaults)
    oot_defaults[['id', 'member_id', 'actual_ead_pct', 'pred_ead_pct']].to_csv("outputs/scorecards/ead_predictions.csv", index=False)
    ead_model.generate_report(train_df, oot_df, "outputs/reports/ead_model_report.pdf", metrics_ead)
    
    # Verify ECL Engine batch run
    from ecl_engine import ECLEngine
    ecl_engine = ECLEngine()
    oot_ecl_df = ecl_engine.predict_batch_ecl(oot_df)
    print(f"OOT Average Expected Credit Loss (ECL): ${oot_ecl_df['ecl'].mean():.2f}")
    
    # 9. PDF Reports Generation
    print("\n9. Generating Institutional PDF Reports...")
    pdf_gen = PDFReportGenerator()
    pdf_gen.generate_validation_report(
        "outputs/reports/validation_report.pdf",
        champion_metrics, challenger_metrics, decile_df,
        roc_img, ks_img, cal_img, psi_val
    )
    
    # Generate XGBoost feature importance report
    importance_df = xgb_challenger.get_feature_importance()
    pdf_gen.generate_feature_importance_report(
        "outputs/reports/feature_importance_report.pdf",
        importance_df,
        shap_png
    )
    
    pdf_gen.generate_policy_report(
        "outputs/reports/policy_recommendations.pdf",
        recommend, cutoff_sim, tradeoff_img, profit_img
    )
    
    pdf_gen.generate_monitoring_report(
        "outputs/reports/score_monitoring_report.pdf",
        score_stats, migration_matrix, psi_val, csi_vals,
        dist_img, band_img
    )
    
    pdf_gen.generate_model_documentation(
        "Model_Documentation.pdf",
        woe_model.mappings, champion_metrics, psi_val
    )
    print("PDF reports generated successfully.")

    # 10. PPTX Presentation Generation
    print("\n10. Generating Executive PowerPoint Presentation...")
    ppt_gen = PPTPresentationGenerator()
    recommend['psi'] = psi_val
    # Load optimal values from decision metrics
    bal_cutoff = recommend['balanced']['cutoff']
    bal_row = cutoff_sim[cutoff_sim['cutoff'] == bal_cutoff].iloc[0]
    recommend['balanced']['approved_loans'] = int(bal_row['approved_loans'])
    recommend['balanced']['approved_amount'] = float(bal_row['approved_amount'])
    recommend['balanced']['expected_revenue'] = float(bal_row['expected_revenue'])
    recommend['balanced']['expected_loss'] = float(bal_row['expected_loss'])
    recommend['balanced']['net_margin_pct'] = float(bal_row['net_margin_pct'])
    
    ppt_gen.generate_presentation(
        "Business_Summary.pptx",
        champion_metrics, recommend
    )
    print("PowerPoint presentation generated successfully.")
    
    print("\n==================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("All models, reports, and documentation have been generated.")
    print("==================================================")

if __name__ == "__main__":
    main()
