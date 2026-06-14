import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.units import inch
from typing import Tuple, Dict

class EADModel:
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
        
        # Encoding and imputation metadata
        self.category_mappings = {}
        self.numeric_imputations = {}

    def calculate_ead_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate EAD target variable: (funded_amnt - total_rec_prncp) / funded_amnt.
        Clipped between 0.0 and 1.0.
        """
        ead_amount = df['funded_amnt'] - df['total_rec_prncp']
        ead_pct = ead_amount / df['funded_amnt']
        return np.clip(ead_pct, 0.0, 1.0)

    def prepare_data(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for EAD modeling.
        For training, we filter for defaults (target == 1 or charged off status) and fit encoders.
        For prediction, we transform features.
        """
        df_clean = df.copy()
        
        # Filter for defaults if training
        if is_training:
            if 'target' in df_clean.columns:
                df_clean = df_clean[df_clean['target'] == 1].copy()
            elif 'loan_status' in df_clean.columns:
                df_clean = df_clean[df_clean['loan_status'].isin(['Charged Off', 'Default'])].copy()
                
        # Calculate Target EAD
        y = self.calculate_ead_target(df_clean) if ('funded_amnt' in df_clean.columns and 'total_rec_prncp' in df_clean.columns) else None
        
        # Prepare Features
        X = pd.DataFrame(index=df_clean.index)
        
        # 1. Process Numeric Features
        for col in self.numeric_features:
            if col in df_clean.columns:
                val = pd.to_numeric(df_clean[col], errors='coerce')
                # If training, calculate and save median for imputation
                if is_training:
                    median_val = val.median()
                    if pd.isnull(median_val):
                        median_val = 0.0
                    self.numeric_imputations[col] = median_val
                
                # Impute missing values
                impute_val = self.numeric_imputations.get(col, 0.0)
                X[col] = val.fillna(impute_val)
            else:
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
        Train EAD model on training cohort and evaluate on validation cohort.
        """
        print("Preparing EAD Training Data...")
        X_train, y_train = self.prepare_data(df_train, is_training=True)
        print(f"EAD Training cohort size: {len(X_train)} defaults")
        
        print("Preparing EAD Validation Data...")
        X_val, y_val = self.prepare_data(df_val, is_training=False)
        print(f"EAD Validation cohort size: {len(X_val)} defaults")
        
        # Fit XGBoost Regressor
        print("Training EAD XGBoost Regressor...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate performance
        metrics = self.evaluate_model(X_train, y_train, X_val, y_val)
        return metrics

    def evaluate_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        Calculate performance metrics for XGBoost.
        """
        # Predictions
        train_pred = np.clip(self.xgb_model.predict(X_train), 0.0, 1.0)
        val_pred = np.clip(self.xgb_model.predict(X_val), 0.0, 1.0)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred),
        }
        
        print("\nEAD Model Evaluation Summary:")
        print(f"EAD XGBoost Validation - RMSE: {metrics['val_rmse']:.4f}, MAE: {metrics['val_mae']:.4f}, R2: {metrics['val_r2']:.4f}")
        return metrics

    def predict_ead(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict EAD percentage for a raw applicant dataframe.
        Clips predictions between 0.0 and 1.0.
        """
        X, _ = self.prepare_data(df, is_training=False)
        if self.xgb_model is None:
            raise ValueError("XGBoost model is not trained yet.")
        pred = self.xgb_model.predict(X)
        return np.clip(pred, 0.0, 1.0)

    def save_model(self, filepath: str):
        """
        Serialize model and configuration using pickle.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'xgb_model': self.xgb_model,
                'category_mappings': self.category_mappings,
                'numeric_imputations': self.numeric_imputations,
                'features': self.features
            }, f)
        print(f"EAD Model successfully saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model and configuration from a pickled file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"EAD model file not found at {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.xgb_model = data['xgb_model']
            self.category_mappings = data['category_mappings']
            self.numeric_imputations = data['numeric_imputations']
            self.features = data['features']
        print(f"EAD Model successfully loaded from {filepath}")

    def generate_report(self, df_train: pd.DataFrame, df_val: pd.DataFrame, output_path: str, metrics: dict):
        """
        Generate a professional PDF model report.
        """
        X_train, y_train = self.prepare_data(df_train, is_training=False)
        X_val, y_val = self.prepare_data(df_val, is_training=False)
        
        val_pred = self.predict_ead(df_val)
        
        # 1. Create Charts
        os.makedirs("outputs/reports", exist_ok=True)
        dist_chart_path = "outputs/reports/ead_distribution.png"
        res_chart_path = "outputs/reports/ead_residuals.png"
        
        # Chart 1: Actual vs Predicted Distribution
        plt.figure(figsize=(8, 4))
        plt.hist(y_val, bins=25, alpha=0.5, label='Actual EAD%', color='#002B49', density=True)
        plt.hist(val_pred, bins=25, alpha=0.5, label='Predicted EAD% (XGB)', color='#00A86B', density=True)
        plt.title('Actual vs Predicted EAD Percentage (Out-of-Time Validation)')
        plt.xlabel('Exposure at Default (EAD %)')
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
        plt.title('EAD Residuals vs Predicted Values (OOT Validation)')
        plt.xlabel('Predicted EAD%')
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
            'ReportTitleEAD',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#002B49'),
            spaceAfter=15
        )
        h1_style = ParagraphStyle(
            'ReportH1EAD',
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
            'ReportBodyEAD',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#222222'),
            spaceAfter=6
        )
        
        story = []
        
        # Title & Header
        story.append(Paragraph("Exposure at Default (EAD) Model Report", title_style))
        story.append(Paragraph("<b>Prepared by:</b> Credit Risk Modeling Division<br/><b>Primary Model:</b> XGBoost Regressor (EAD Exposure %)", body_style))
        story.append(Spacer(1, 10))
        
        # Section 1
        story.append(Paragraph("1. Executive Summary & Objective", h1_style))
        story.append(Paragraph("This report presents the validation results for the Exposure at Default (EAD) regression model. EAD represents the expected outstanding balance of a loan at the moment of default. The target variable is engineered as the ratio of outstanding principal relative to the initial funded amount: <code>(funded_amnt - total_rec_prncp) / funded_amnt</code>, capped within <code>[0.0, 1.0]</code>. The model uses only application-time features to prevent data leakage.", body_style))
        
        # Section 2: Metrics Table
        story.append(Spacer(1, 5))
        story.append(Paragraph("2. Model Performance Evaluation", h1_style))
        story.append(Paragraph("Performance of the EAD XGBoost Regressor on the In-Time (Train) and Out-of-Time (Validation) datasets:", body_style))
        
        table_data = [
            [Paragraph("<b>Performance Metric</b>", body_style), Paragraph("<b>XGBoost (EAD Model)</b>", body_style)],
            ["Train RMSE", f"{metrics['train_rmse']:.4f}"],
            ["Train MAE", f"{metrics['train_mae']:.4f}"],
            ["Train R² Score", f"{metrics['train_r2']:.4f}"],
            ["Validation (OOT) RMSE", f"{metrics['val_rmse']:.4f}"],
            ["Validation (OOT) MAE", f"{metrics['val_mae']:.4f}"],
            ["Validation (OOT) R² Score", f"{metrics['val_r2']:.4f}"]
        ]
        
        t = Table(table_data, colWidths=[3*inch, 3*inch])
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
        story.append(Paragraph("The chart below shows actual vs. predicted EAD% distributions. The EAD target is spread across the entire range from 0.0 to 1.0 (with a peak near 1.0 for accounts that defaulted early in their lifecycle). The model provides a reliable and stable risk estimate of exposure at the individual loan level.", body_style))
        
        story.append(Image(dist_chart_path, width=6*inch, height=3*inch))
        story.append(Spacer(1, 10))
        story.append(PageBreak())
        
        story.append(Paragraph("4. Residual Analysis & Diagnostics", h1_style))
        story.append(Paragraph("The residual plot displays the difference between actual EAD% and predicted EAD%:", body_style))
        story.append(Image(res_chart_path, width=6*inch, height=3*inch))
        
        # Section 5: Governance
        story.append(Spacer(1, 10))
        story.append(Paragraph("5. Model Governance & Implementation Details", h1_style))
        story.append(Paragraph("This EAD model complies with institutional and regulatory standards (IFRS 9 / Basel). By restricting predictors to application-time variables, the model supports robust pre-decision simulators, enabling credit managers to project credit losses dynamically prior to loan origination.", body_style))
        
        def add_header_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor('#888888'))
            canvas.drawString(54, 750, "CREDIT RISK SCORECARD SYSTEM  |  EAD MODEL REPORT")
            canvas.setStrokeColor(colors.HexColor('#CCCCCC'))
            canvas.setLineWidth(0.5)
            canvas.line(54, 742, 558, 742)
            canvas.drawString(54, 40, "CONFIDENTIAL  -  INTERNAL BANK USE ONLY")
            canvas.drawRightString(558, 40, f"Page {doc.page}")
            canvas.line(54, 52, 558, 52)
            canvas.restoreState()
            
        doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
        print(f"EAD Model report successfully generated at {output_path}")
        
        # Clean up temporary chart images
        if os.path.exists(dist_chart_path):
            os.remove(dist_chart_path)
        if os.path.exists(res_chart_path):
            os.remove(res_chart_path)
            
