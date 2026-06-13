# 🏦 End-to-End Credit Risk Decisioning & Probability of Default (PD) Modeling Framework

**An institutional-grade credit scorecard, validation suite, and policy optimization engine.**

*Designed & Developed by Anam*  
*Machine Learning | Risk Analytics | Business Strategy*

---

## 1. Business Problem

Commercial banks and fintech lenders handle thousands of consumer loan applications daily. The primary credit decisioning objective is to optimize the risk-return trade-off: approving creditworthy borrowers to maximize interest income while declining high-risk applicants to minimize default write-offs. 

Subjective, manual underwriting is slow, inconsistent, and highly susceptible to bias. Furthermore, regulatory frameworks (such as the Fair Credit Reporting Act (FCRA), Equal Credit Opportunity Act (ECOA), and Basel II/III Accords) mandate that credit decisioning models must be:
1. **Fully Auditable:** Every credit decision must have clear, mathematically traceable reasons.
2. **Transparent:** Standard credit scores must be explainable to applicants (reason codes).
3. **Calibrated and Stable:** Predicted default probabilities must match empirical default rates over time to support capital adequacy requirements (Expected Loss pricing).

This repository implements a complete, production-grade credit decisioning framework that translates statistical risk modeling into automated credit scoring, policy simulation, and capital provisioning.

---

## 2. Project Overview

This framework establishes a rigorous, end-to-end analytical pipeline that maps raw borrower characteristics to automated credit decisions:
* **Interpretability First:** A **Logistic Regression** model is fitted on **Weight of Evidence (WoE)** binned variables, converting model coefficients into an integer-based **Credit Scorecard (300-850)**.
* **Predictive Rigor:** The scorecard is validated against an **XGBoost Challenger** model on an **Out-of-Time (OOT)** cohort.
* **Risk Governance:** Stability monitoring checks population shifts via **Population Stability Index (PSI)** and **Characteristic Stability Index (CSI)**.
* **Lending Policy Simulation:** A business policy optimizer simulations portfolio net margins, approval ratios, and write-offs to identify the optimal credit cutoff score.
* **Enterprise Decision Portal:** An interactive dashboard designed for risk officers, credit auditors, and business stakeholders.

---

## 3. Architecture

The system utilizes a modular python library structure:

```
├── configs/
│   └── config.yaml                  # Scaling (PDO, Base Odds) & Financial parameters (LGD, Rate)
├── data/
│   └── loan.csv                     # Historical Lending Club consumer credit database
├── src/
│   ├── data_processing.py           # Ingestion, label definition, and temporal splitting
│   ├── woe_binning.py               # Decision-tree binned WoE transformer
│   ├── scorecard.py                 # Logistic regression fitting & Scorecard point scaling
│   ├── xgboost_model.py             # Challenger tree-boosting model training
│   ├── validation.py                # ROC-AUC, KS separation, and Brier Score calibration
│   ├── stability.py                 # PSI and CSI population drift tracking
│   ├── score_monitoring.py          # Score distribution shifts & migration matrices
│   ├── business_decision_engine.py  # Portfolio margin & ECL simulations
│   ├── pdf_generator.py             # Enterprise PDF risk reports compiler
│   └── ppt_generator.py             # Executive PowerPoint decks compiler
├── dashboard/
│   └── app.py                       # Enterprise Credit Decision Portal (Streamlit)
├── tests/
│   └── test_score_engine.py         # Unit tests validating scorecard points lookup math
├── verify_pipeline.py               # Central execution orchestrator
└── score_generation_engine.py       # Production scoring module
```

---

## 4. Data Methodology

### 4.1. Label Definition & Cohort Segmentation
The model is trained on historical retail lending data (39,717 applications). 
* **Target Label:** Binary default classification target. **Default (1)** represents loans categorized as `Charged Off` or `Default`. **Non-default (0)** represents `Fully Paid` loans. Active loans (`Current`) are excluded.
* **Temporal Cohorts:** To prevent temporal leakage and simulate real production deployments, the dataset is split chronologically:
  * **In-Time (Train/Test) (18,061 loans):** Loans issued from 2007 through 2010.
  * **Out-of-Time (OOT) Validation (20,516 loans):** Loans issued in 2011. This cohort validates model performance against a completely separate temporal group.

### 4.2. WoE Binning & Information Value (IV) Selection
Numeric parameters are grouped into coarse risk bins using a Decision Tree segmentation algorithm. Categorical metrics are grouped by default rate similarities. Weight of Evidence (WoE) is computed as:

$$\text{WoE}_i = \ln\left( \frac{\%\text{ Non-Defaults}_i}{\%\text{ Defaults}_i} \right)$$

Features are screened using the Information Value (IV) metric:

$$\text{IV} = \sum \left( \%\text{ Non-Defaults}_i - \%\text{ Defaults}_i \right) \times \text{WoE}_i$$

* **Variables Retained ($0.02 \le IV \le 0.50$):** `revol_util`, `purpose`, `inq_last_6mths`, `annual_inc`, `pub_rec`, `pub_rec_bankruptcies`.
* **Variables Excluded (IV < 0.02 - Useless/Policy-Only):** `loan_amnt`, `dti`, `open_acc`, `credit_history_age`, `emp_length`, `home_ownership`, `delinq_2yrs`. These are collected for underwriting rules but carry zero scorecard points.

---

## 5. Modeling Approach

### 5.1. Champion Scorecard Scaling
The Champion model utilizes a Logistic Regression classifier fitted on WoE-transformed inputs. The model log-odds are scaled linearly into a standard banking scorecard:

$$\text{Score} = \text{Factor} \times \ln(\text{Odds}) + \text{Offset}$$

Scaling parameters configured in `configs/config.yaml`:
* **Base Score:** 600 points at Base Odds of 50:1 (Good-to-Bad ratio).
* **Points to Double Odds (PDO):** 20.
* **Score Bounds:** 300 to 850 points.

### 5.2. Challenger XGBoost Model
A non-linear gradient-boosted decision tree classifier (XGBoost) is trained on raw inputs, serving as a benchmark to assess predictive loss from linear binning.

### 5.3. Reject Inference Framework
Because the modeling dataset only includes approved and booked loans, it suffers from **sample selection bias** (we cannot observe default rates of rejected applicants). In next-generation releases, the bank will implement **Reject Inference**:
1. **Fuzzy Augmentation:** Apply the model to rejected applications to predict default probabilities. Each rejected record is duplicated into "fractional good" and "fractional bad" records and re-weighted.
2. **Parceling:** Categorize rejected applicants into risk bins. Impute default outcomes to a sample of each bin matching approved default rates scaled up by a risk factor (e.g., 2.0x).
3. **Re-weighting:** Assign weights to approved applicants to represent the broader applicant pool, inflating the importance of approved applicants who resemble rejected individuals.

---

## 6. Validation & Performance Results

Model metrics evaluated on the Out-of-Time (OOT) cohort:

| Validation Metric | Champion Scorecard (Logistic) | Challenger XGBoost | Advantage / Business Context |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | 0.6311 | 0.6331 | Challenger has a marginal 0.20% separation lift. |
| **KS Statistic** | 0.1870 | 0.1991 | Both models demonstrate solid risk group separation. |
| **Gini Coefficient** | 0.2622 | 0.2662 | Scorecard maintains 98.5% of XGBoost's predictive power. |
| **Brier Score** | **0.1308** | 0.2274 | **Champion Scorecard is significantly better calibrated** (closer to 0). |

*Note: While XGBoost shows a minor improvement in ROC-AUC, the Scorecard is retained due to regulatory transparency requirements and superior probability calibration (Brier Score of 0.1308 vs. 0.2274 for XGBoost).*

### 6.1. Stability & Population Drift
* **Population Stability Index (PSI):** **0.0091** (significantly below 0.10, indicating **Stable** population risk distributions between train and OOT validation sets).
* **Characteristic Drift (CSI):** All input variables remain highly stable, except for `pub_rec_bankruptcies` (CSI = 0.2860), which flags population drift due to shifting consumer bankruptcy profiles in 2011.

---

## 7. Business & Policy Impact

By simulating various score cutoff thresholds, risk executives can manage portfolio performance:
* **Optimal Score Cutoff:** **620**
* **Projected Approval Rate:** **90.8%**
* **Expected Portfolio Default Rate:** **14.60%** (down from a raw baseline default rate of 15.87% on OOT validation)
* **Approved Loan Volume:** **$222.96M** (out of $240.94M requested)
* **Projected Portfolio Net Profit Yield:** **$65.04M** (optimized interest yields minus expected default write-offs).

---

## 8. Dashboard Demo

The interactive **Enterprise Credit Decision Portal** (`dashboard/app.py`) is organized into 7 distinct risk modules:
1. **Executive Overview:** High-level executive briefing containing the 6 primary credit risk KPIs (Total Loans, Default Rate, ROC-AUC, KS, Gini, PSI) and portfolio distribution.
2. **Credit Risk Simulator:** Production underwriting tool. Input borrower parameters, view the automated decision (Approve/Refer/Decline), inspect FICO placement on a gauge indicator, and view the transparent scorecard point breakdown. Shows clear segregation between Model Predictors and Policy Info.
3. **Project Journey:** Vertical step-by-step visual stepper outlining data pipeline stages from source ingestion to policy cutoff.
4. **Layman's Explainer:** A non-technical guide explaining credit risk concepts (WoE, IV, scorecards) to recruiters and MBA interviewers.
5. **Model Performance:** Side-by-side performance curves (ROC, KS, Probability Calibration) comparing Champion and Challenger.
6. **Risk Monitoring:** Population stability tracking, displaying PSI matrices, characteristic drift (CSI) status, and score distribution shifts.
7. **Lending Policy Simulator:** Cutoff optimizer. Adjust score cutoffs using a slider to view dynamic approval rates, defaults, expected credit losses, and revenue yields on profit curves.

---

## 9. Future Improvements

To enhance framework capabilities, subsequent development cycles will incorporate:
* **Macroeconomic Overlay Factors:** Adjusting scorecard probabilities of default (PD) using macroeconomic indicators (e.g. unemployment rate, interest rates, CPI) to compute point-in-time (PIT) PDs.
* **Machine Learning Calibration (XGBoost):** Utilizing isotonic regression or Platt scaling on the Challenger XGBoost model to improve Brier Score calibration, preparing it for champion deployment.
* **Automated Re-Binning Triggers:** Ingesting live monitoring CSI values to flag drifted variables (like bankruptcies) and trigger automatic re-binning and model re-training pipelines.

---

*Designed & Developed by Anam*  
*Machine Learning | Risk Analytics | Business Strategy*
