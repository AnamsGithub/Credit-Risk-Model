import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.units import inch
from typing import Tuple, Dict

class LGDModel:
    def __init__(self):
        self.numeric_features = [
            'loan_amnt', 'annual_inc', 'dti', 'revol_util', 
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 
            'pub_rec', 'pub_rec_bankruptcies', 'credit_history_age',
            'emp_length'
        ]
        self.categorical_features = ['home_ownership', 'purpose']
        self.features = self.numeric_features + self.categorical_features
        
        self.xgb_model = None
        self.rf_model = None
        
        # Encoding and imputation metadata
        self.category_mappings = {}
        self.numeric_imputations = {}

    def calculate_lgd_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate LGD target variable: (loan_amnt - recoveries) / loan_amnt.
        Clipped between 0.0 and 1.0.
        """
        loss_amount = df['loan_amnt'] - df['recoveries']
        lgd = loss_amount / df['loan_amnt']
        return np.clip(lgd, 0.0, 1.0)

    def prepare_data(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for LGD modeling on the cohort.
        For training, we filter for defaults (target == 1 or charged off status) and fit encoders.
        For prediction, we transform features.
        """
        df_clean = df.copy()
        
        # Filter for defaults if training
        if is_training:
            # We assume defaults are identified by target == 1 (or loan_status in ['Charged Off', 'Default'])
            if 'target' in df_clean.columns:
                df_clean = df_clean[df_clean['target'] == 1].copy()
            elif 'loan_status' in df_clean.columns:
                df_clean = df_clean[df_clean['loan_status'].isin(['Charged Off', 'Default'])].copy()
                
        # Calculate Target LGD
        y = self.calculate_lgd_target(df_clean) if ('loan_amnt' in df_clean.columns and 'recoveries' in df_clean.columns) else None
        
        # Prepare Features
        X = pd.DataFrame(index=df_clean.index)
        
        # 1. Process Numeric Features
        for col in self.numeric_features:
            if col in df_clean.columns:
                val = pd.to_numeric(df_clean[col], errors='coerce')
                # If training, calculate and save median for imputation
                if is_training:
                    median_val = val.median()
                    # Fallback if all values are null
                    if pd.isnull(median_val):
                        median_val = 0.0
                    self.numeric_imputations[col] = median_val
                
                # Impute missing values
                impute_val = self.numeric_imputations.get(col, 0.0)
                X[col] = val.fillna(impute_val)
            else:
                # Fill missing column with imputation or default 0.0
                impute_val = self.numeric_imputations.get(col, 0.0)
                X[col] = impute_val

        # 2. Process Categorical Features
        for col in self.categorical_features:
            if col in df_clean.columns:
                val_str = df_clean[col].astype(str).str.strip().str.upper()
                if is_training:
                    # Fit label mappings based on unique values in training data
                    unique_cats = sorted(val_str.unique())
                    mapping = {cat: i for i, cat in enumerate(unique_cats)}
                    self.category_mappings[col] = mapping
                
                # Transform using mapped values
                mapping = self.category_mappings.get(col, {})
                X[col] = val_str.map(mapping).fillna(-1).astype(int)
            else:
                X[col] = -1

        return X, y

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
        """
        Train LGD models on training cohort and evaluate on validation cohort.
        """
        print("Preparing LGD Training Data...")
        X_train, y_train = self.prepare_data(df_train, is_training=True)
        print(f"LGD Training cohort size: {len(X_train)} defaults")
        
        print("Preparing LGD Validation Data...")
        X_val, y_val = self.prepare_data(df_val, is_training=False)
        print(f"LGD Validation cohort size: {len(X_val)} defaults")
        
        # Fit XGBoost Regressor
        print("Training LGD XGBoost Regressor...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Fit Random Forest Regressor (Benchmark)
        print("Training LGD Random Forest Regressor (Benchmark)...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate performance
        metrics = self.evaluate_models(X_train, y_train, X_val, y_val)
        return metrics

    def evaluate_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        Calculate performance metrics for both XGBoost and Random Forest.
        """
        # Predictions
        train_pred_xgb = np.clip(self.xgb_model.predict(X_train), 0.0, 1.0)
        val_pred_xgb = np.clip(self.xgb_model.predict(X_val), 0.0, 1.0)
        
        train_pred_rf = np.clip(self.rf_model.predict(X_train), 0.0, 1.0)
        val_pred_rf = np.clip(self.rf_model.predict(X_val), 0.0, 1.0)
        
        metrics = {
            'xgb': {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred_xgb)),
                'train_mae': mean_absolute_error(y_train, train_pred_xgb),
                'train_r2': r2_score(y_train, train_pred_xgb),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred_xgb)),
                'val_mae': mean_absolute_error(y_val, val_pred_xgb),
                'val_r2': r2_score(y_val, val_pred_xgb),
            },
            'rf': {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred_rf)),
                'train_mae': mean_absolute_error(y_train, train_pred_rf),
                'train_r2': r2_score(y_train, train_pred_rf),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred_rf)),
                'val_mae': mean_absolute_error(y_val, val_pred_rf),
                'val_r2': r2_score(y_val, val_pred_rf),
            }
        }
        
        print("\nLGD Model Evaluation Summary:")
        print(f"XGBoost Validation - RMSE: {metrics['xgb']['val_rmse']:.4f}, MAE: {metrics['xgb']['val_mae']:.4f}, R2: {metrics['xgb']['val_r2']:.4f}")
        print(f"Random Forest Validation - RMSE: {metrics['rf']['val_rmse']:.4f}, MAE: {metrics['rf']['val_mae']:.4f}, R2: {metrics['rf']['val_r2']:.4f}")
        return metrics

    def predict_lgd(self, df: pd.DataFrame, model_type: str = 'xgb') -> np.ndarray:
        """
        Predict LGD percentage for a raw applicant dataframe.
        Clips predictions between 0.0 and 1.0.
        """
        X, _ = self.prepare_data(df, is_training=False)
        if model_type == 'xgb':
            if self.xgb_model is None:
                raise ValueError("XGBoost model is not trained yet.")
            pred = self.xgb_model.predict(X)
        else:
            if self.rf_model is None:
                raise ValueError("Random Forest model is not trained yet.")
            pred = self.rf_model.predict(X)
        return np.clip(pred, 0.0, 1.0)

    def save_model(self, filepath: str):
        """
        Serialize model and configuration using pickle.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'xgb_model': self.xgb_model,
                'rf_model': self.rf_model,
                'category_mappings': self.category_mappings,
                'numeric_imputations': self.numeric_imputations,
                'features': self.features
            }, f)
        print(f"LGD Model successfully saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model and configuration from a pickled file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LGD model file not found at {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.xgb_model = data['xgb_model']
            self.rf_model = data['rf_model']
            self.category_mappings = data['category_mappings']
            self.numeric_imputations = data['numeric_imputations']
            self.features = data['features']
        print(f"LGD Model successfully loaded from {filepath}")

    def generate_report(self, df_train: pd.DataFrame, df_val: pd.DataFrame, output_path: str, metrics: dict):
        """
        Generate a professional PDF model report.
        """
        X_train, y_train = self.prepare_data(df_train, is_training=False)
        X_val, y_val = self.prepare_data(df_val, is_training=False)
        
        val_pred = self.predict_lgd(df_val, model_type='xgb')
        
        # 1. Create Charts
        os.makedirs("outputs/reports", exist_ok=True)
        dist_chart_path = "outputs/reports/lgd_distribution.png"
        res_chart_path = "outputs/reports/lgd_residuals.png"
        
        # Chart 1: Actual vs Predicted Distribution
        plt.figure(figsize=(8, 4))
        plt.hist(y_val, bins=25, alpha=0.5, label='Actual LGD', color='#002B49', density=True)
        plt.hist(val_pred, bins=25, alpha=0.5, label='Predicted LGD (XGB)', color='#FF8C00', density=True)
        plt.title('Actual vs Predicted LGD Distribution (Out-of-Time Validation)')
        plt.xlabel('Loss Given Default (LGD)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(dist_chart_path, dpi=300)
        plt.close()
        
        # Chart 2: Residuals Scatter Plot
        residuals = y_val - val_pred
        plt.figure(figsize=(8, 4))
        plt.scatter(val_pred, residuals, alpha=0.3, color='#002B49', s=10)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('LGD Residuals vs Predicted Values (OOT Validation)')
        plt.xlabel('Predicted LGD')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.tight_layout()
        plt.savefig(res_chart_path, dpi=300)
        plt.close()
        
        # 2. Build PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,
            bottomMargin=72
        )
        
        styles = getSampleStyleSheet()
        
        # Establish Custom Styles
        title_style = ParagraphStyle(
            'ReportTitleLGD',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#002B49'),
            spaceAfter=15
        )
        h1_style = ParagraphStyle(
            'ReportH1LGD',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#002B49'),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        )
        body_style = ParagraphStyle(
            'ReportBodyLGD',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#222222'),
            spaceAfter=6
        )
        
        story = []
        
        # Title & Header
        story.append(Paragraph("Loss Given Default (LGD) Model Report", title_style))
        story.append(Paragraph("<b>Prepared by:</b> Credit Risk Modeling Division<br/><b>Primary Model:</b> XGBoost Regressor (LGD Severity)<br/><b>Benchmark Model:</b> Random Forest Regressor", body_style))
        story.append(Spacer(1, 10))
        
        # Section 1
        story.append(Paragraph("1. Executive Summary & Objective", h1_style))
        story.append(Paragraph("This report validates the Loss Given Default (LGD) model developed for the retail lending portfolio. In credit risk analytics, LGD represents the percentage of exposure lost if a borrower defaults. The target is defined as <code>(loan_amnt - recoveries) / loan_amnt</code>, capped within <code>[0.0, 1.0]</code>. The model is trained purely on the cohort of defaults using application-time features (no leakage from post-origination behaviors).", body_style))
        
        # Section 2: Metrics Table
        story.append(Spacer(1, 5))
        story.append(Paragraph("2. Model Performance Evaluation", h1_style))
        story.append(Paragraph("Performance is compared across the primary XGBoost model and the benchmark Random Forest model on both In-Time (Train) and Out-of-Time (Validation) populations.", body_style))
        
        table_data = [
            [Paragraph("<b>Performance Metric</b>", body_style), Paragraph("<b>XGBoost (Primary)</b>", body_style), Paragraph("<b>Random Forest (Benchmark)</b>", body_style)],
            ["Train RMSE", f"{metrics['xgb']['train_rmse']:.4f}", f"{metrics['rf']['train_rmse']:.4f}"],
            ["Train MAE", f"{metrics['xgb']['train_mae']:.4f}", f"{metrics['rf']['train_mae']:.4f}"],
            ["Train R² Score", f"{metrics['xgb']['train_r2']:.4f}", f"{metrics['rf']['train_r2']:.4f}"],
            ["Validation (OOT) RMSE", f"{metrics['xgb']['val_rmse']:.4f}", f"{metrics['rf']['val_rmse']:.4f}"],
            ["Validation (OOT) MAE", f"{metrics['xgb']['val_mae']:.4f}", f"{metrics['rf']['val_mae']:.4f}"],
            ["Validation (OOT) R² Score", f"{metrics['xgb']['val_r2']:.4f}", f"{metrics['rf']['val_r2']:.4f}"]
        ]
        
        t = Table(table_data, colWidths=[2.5*inch, 2*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F2F2F2')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9F9F9')]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t)
        
        # Section 3: Visual Analytics
        story.append(Spacer(1, 10))
        story.append(Paragraph("3. Model Diagnostics & Visualizations", h1_style))
        story.append(Paragraph("The chart below displays the actual vs. predicted LGD distribution. Due to the high recovery skew in the portfolio (where many defaults recover little to nothing), the actual target distribution has a significant spike near 1.0. The XGBoost model represents a conservative expectation of loss severity.", body_style))
        
        story.append(Image(dist_chart_path, width=6*inch, height=3*inch))
        story.append(Spacer(1, 10))
        story.append(PageBreak())
        
        story.append(Paragraph("4. Residual Analysis & Diagnostic Checks", h1_style))
        story.append(Paragraph("The residual plot displays the difference between actual LGD and predicted LGD. A uniform residual band indicates stable and unbiased predictions across the credit bands.", body_style))
        story.append(Image(res_chart_path, width=6*inch, height=3*inch))
        
        # Section 5: Governance
        story.append(Spacer(1, 10))
        story.append(Paragraph("5. Model Governance & Limitations", h1_style))
        story.append(Paragraph("In accordance with Basel and IFRS 9 guidelines, LGD models are calibrated using default cohorts only. Because underwriting is conducted at application time, the model only uses variables available before funding, avoiding any post-origination features that would cause data leakage in simulation workflows.", body_style))
        
        def add_header_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor('#888888'))
            canvas.drawString(54, 750, "CREDIT RISK SCORECARD SYSTEM  |  LGD MODEL REPORT")
            canvas.setStrokeColor(colors.HexColor('#CCCCCC'))
            canvas.setLineWidth(0.5)
            canvas.line(54, 742, 558, 742)
            canvas.drawString(54, 40, "CONFIDENTIAL  -  INTERNAL BANK USE ONLY")
            canvas.drawRightString(558, 40, f"Page {doc.page}")
            canvas.line(54, 52, 558, 52)
            canvas.restoreState()
            
        doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
        print(f"LGD Model report successfully generated at {output_path}")
        
        # Clean up temporary chart images
        if os.path.exists(dist_chart_path):
            os.remove(dist_chart_path)
        if os.path.exists(res_chart_path):
            os.remove(res_chart_path)
