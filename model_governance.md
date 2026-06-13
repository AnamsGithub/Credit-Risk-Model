# Model Risk Management & Governance Framework
**Retail Credit Risk PD Scorecard Model v1.0**

This document details the model governance, risk management policies, operational controls, and validation procedures required by the bank's Model Risk Management (MRM) department to ensure compliance with institutional and regulatory (Basel/SR 11-7) guidelines.

---

## 1. Key Modeling Assumptions

To ensure model validity and reliability, several foundational assumptions are established and must be periodically reviewed:

### 1.1. Representative Behavior (Historical Representation)
* **Assumption:** The historical loan data from the Lending Club dataset is representative of future borrower cohorts and credit profiles.
* **Risk:** Shifts in customer sourcing strategies, underwriting standards, or marketing channels may lead to population drift, rendering historical defaults non-representative.

### 1.2. Macroeconomic Stability
* **Assumption:** The macroeconomic conditions during the training period (inflation, interest rates, employment rates) remain stable.
* **Risk:** Severe economic contractions, high inflation, or labor market disruptions will increase systemic defaults, violating baseline PD calibrations.

### 1.3. Consumer Credit Trends
* **Assumption:** The correlation between borrower credit characteristics (e.g., DTI, inquiries, revolving utilization) and default behavior remains consistent over time.
* **Risk:** Changes in consumer payment preferences, alternative credit availability, or debt-restructuring trends might alter characteristic correlations.

---

## 2. Model Limitations & Bias

### 2.1. Reject Inference & Sample Selection Bias
* **Limitation:** The model is trained exclusively on approved and booked loans (observed performance). It does not observe the performance of rejected applicants, creating sample selection bias.
* **Mitigation:** Future model revisions should implement reject inference methodologies (Fuzzy Augmentation or Parceling) to impute outcomes for rejected applicants and re-weight the modeling sample.

### 2.2. Data Scope and Recency
* **Limitation:** The dataset represents prime and near-prime peer-to-peer retail borrowers from the historical Lending Club cohort. It may not generalize to commercial lending, auto loans, or secured mortgage credit products.

---

## 3. Model Monitoring Frequency

Consistent monitoring is required to verify that the scoring engine remains accurate and stable:

| Frequency | Monitoring Scope | Responsible Unit |
| :--- | :--- | :--- |
| **Monthly** | Score Distribution Tracking, Population Stability Index (PSI), System Delinquency Rates | Credit Risk Team / Operations |
| **Quarterly** | Characteristic Stability Index (CSI) tracking, Gini/KS Performance checks, Migration Matrices | Independent Model Validation Unit |
| **Annually** | Comprehensive Redevelopment Review, Parameter recalibrations, Independent Audit | Internal Audit & Risk Committee |

---

## 4. Retraining and Redevelopment Triggers

An automatic alert and remediation process is initiated if monitoring metrics cross defined risk thresholds:

### 4.1. Population Stability Index (PSI) Triggers
* **PSI < 0.10 (Green):** Stable. No action required.
* **0.10 <= PSI < 0.25 (Amber):** Moderate shift. Increase monitoring frequency, analyze CSI values to identify the source of drift, and prepare parameter recalibration plans.
* **PSI >= 0.25 (Red):** Significant drift. Trigger immediate model redevelopment, parameter recalibration, or halt automated approvals for impacted cohorts.

### 4.2. Model Performance Triggers
* **Gini/AUC Degradation:** A drop in the Gini coefficient of > 15% from the validation baseline (or ROC-AUC falling below 0.60) triggers immediate model recalibration.
* **KS Statistic Degradation:** If the KS statistic drops below 0.25, the model's capacity to separate default risk is deemed insufficient, triggering redevelopment.

### 4.3. External Triggers
* **Regulatory Changes:** Revisions to credit scoring guidelines (FCRA, Fair Lending rules, Basel IV capital requirements) will initiate an out-of-cycle redevelopment.

---

## 5. Data Quality Checks & Governance Controls

To prevent data corruption, target leakage, or erroneous credit decisioning:
1. **Target Leakage Audits:** Pipeline configs must block post-origination fields (payments, collections, write-offs) from model input.
2. **Missing Value Controls:** The scoring engine must explicitly handle missing values via defined "Missing" bin categories, preventing application crashes.
3. **Change Management:** All modifications to `configs/config.yaml` or `scorecard_table.csv` must go through a peer-reviewed pull request process and be approved by the Chief Risk Officer (CRO).
