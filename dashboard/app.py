import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import textwrap

# Add paths
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('src'))

from score_generation_engine import ScoreGenerationEngine

# HTML Table Helpers with dynamic light/dark mode CSS variables

def clean_html(html_str):
    return "\n".join([line.strip() for line in html_str.split("\n") if line.strip()])

def make_cc_table_html(df):
    html = """<table style="width:100%; border-collapse: collapse; font-family:'Inter', sans-serif; font-size: 0.9rem; margin-bottom:25px;">
    <thead>
        <tr style="border-bottom: 2px solid var(--secondary-background-color); text-align: left; background-color: var(--secondary-background-color);">
            <th style="padding: 12px 10px; color: var(--text-color);">Model Designator</th>
            <th style="padding: 12px 10px; text-align:right; color: var(--text-color);">ROC-AUC</th>
            <th style="padding: 12px 10px; text-align:right; color: var(--text-color);">KS Statistic</th>
            <th style="padding: 12px 10px; text-align:right; color: var(--text-color);">Gini</th>
            <th style="padding: 12px 10px; text-align:right; color: var(--text-color);">Brier Score</th>
            <th style="padding: 12px 10px; text-align:right; color: var(--text-color);">Precision</th>
            <th style="padding: 12px 10px; text-align:right; color: var(--text-color);">Recall</th>
            <th style="padding: 12px 10px; text-align:right; color: var(--text-color);">F1-Score</th>
        </tr>
    </thead>
    <tbody>"""
    for idx, row in df.iterrows():
        model = row['Model']
        auc = f"{row['ROC-AUC']:.4f}"
        ks = f"{row['KS']:.4f}"
        gini = f"{row['Gini']:.4f}"
        brier = f"{row['Brier Score']:.4f}"
        prec = f"{row['Precision']:.4f}"
        rec = f"{row['Recall']:.4f}"
        f1 = f"{row['F1-Score']:.4f}"
        
        is_champ = "Champion" in model
        row_style = "background-color: var(--background-color); border-left: 4px solid #16A34A;" if is_champ else "border-left: 4px solid #94A3B8;"
        badge = "<span style='color:#0F5132; background-color:#D1E7DD; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:0.75rem; margin-right:6px;'>CHAMPION</span>" if is_champ else "<span style='color:#41464B; background-color:#E2E3E5; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:0.75rem; margin-right:6px;'>CHALLENGER</span>"
        
        html += f"""
        <tr style="border-bottom: 1px solid var(--secondary-background-color); {row_style}">
            <td style="padding: 12px 10px; font-weight: 600; color: var(--text-color);">{badge} {model.replace('Champion ', '').replace('Challenger ', '')}</td>
            <td style="padding: 12px 10px; font-family: monospace; text-align:right; font-weight:bold; color: var(--text-color);">{auc}</td>
            <td style="padding: 12px 10px; font-family: monospace; text-align:right; color: var(--text-color);">{ks}</td>
            <td style="padding: 12px 10px; font-family: monospace; text-align:right; color: var(--text-color);">{gini}</td>
            <td style="padding: 12px 10px; font-family: monospace; text-align:right; font-weight:bold; color: {'#157347' if is_champ else '#dc3545'};">{brier}</td>
            <td style="padding: 12px 10px; font-family: monospace; text-align:right; color: var(--text-color);">{prec}</td>
            <td style="padding: 12px 10px; font-family: monospace; text-align:right; color: var(--text-color);">{rec}</td>
            <td style="padding: 12px 10px; font-family: monospace; text-align:right; color: var(--text-color);">{f1}</td>
        </tr>"""
    html += "</tbody></table>"
    return clean_html(html)

def make_psi_details_html(df):
    html = """<table style="width:100%; border-collapse: collapse; font-family:'Inter', sans-serif; font-size: 0.85rem;">
    <thead>
        <tr style="border-bottom: 2px solid var(--secondary-background-color); text-align: left; background-color: var(--secondary-background-color);">
            <th style="padding: 10px; color: var(--text-color);">Score Range Bin</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">Expected (Train %)</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">Actual (OOT %)</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">PSI Term</th>
        </tr>
    </thead>
    <tbody>"""
    for idx, row in df.iterrows():
        bin_r = row['bin_range']
        exp = f"{row['expected_pct']:.2%}"
        act = f"{row['actual_pct']:.2%}"
        psi = f"{row['psi_term']:.5f}"
        html += f"""
        <tr style="border-bottom: 1px solid var(--secondary-background-color);">
            <td style="padding: 10px; font-family: monospace; color: var(--text-color); font-weight: 500;">{bin_r}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{exp}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{act}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{psi}</td>
        </tr>"""
    html += "</tbody></table>"
    return clean_html(html)

def make_csi_table_html(df):
    html = """<table style="width:100%; border-collapse: collapse; font-family:'Inter', sans-serif; font-size: 0.85rem;">
    <thead>
        <tr style="border-bottom: 2px solid var(--secondary-background-color); text-align: left; background-color: var(--secondary-background-color);">
            <th style="padding: 10px; color: var(--text-color);">Characteristic</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">CSI Value</th>
            <th style="padding: 10px; text-align:center; color: var(--text-color);">Drift Interpretation</th>
        </tr>
    </thead>
    <tbody>"""
    for idx, row in df.iterrows():
        char = row['Characteristic'].replace('_', ' ').title()
        val = f"{row['CSI']:.4f}"
        interp = row['Interpretation']
        
        if "Significant" in interp:
            badge = "<span class='badge-drift'>Significant Drift</span>"
        elif "Moderate" in interp:
            badge = "<span class='badge-moderate'>Moderate Drift</span>"
        else:
            badge = "<span class='badge-stable'>Stable</span>"
            
        html += f"""
        <tr style="border-bottom: 1px solid var(--secondary-background-color);">
            <td style="padding: 10px; font-weight: 500; color: var(--text-color);">{char}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{val}</td>
            <td style="padding: 10px; text-align:center;">{badge}</td>
        </tr>"""
    html += "</tbody></table>"
    return clean_html(html)

def make_mig_table_html(df):
    html = """<table style="width:100%; border-collapse: collapse; font-family:'Inter', sans-serif; font-size: 0.85rem;">
    <thead>
        <tr style="border-bottom: 2px solid var(--secondary-background-color); text-align: left; background-color: var(--secondary-background-color);">
            <th style="padding: 10px; color: var(--text-color);">OOT Cohort Split</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">Excellent</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">Good</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">Moderate</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">High Risk</th>
            <th style="padding: 10px; text-align:right; color: var(--text-color);">Very High</th>
        </tr>
    </thead>
    <tbody>"""
    for idx, row in df.iterrows():
        cohort = idx
        exc = f"{row['1. Excellent']:.2f}%"
        gd = f"{row['2. Good']:.2f}%"
        mod = f"{row['3. Moderate']:.2f}%"
        hr = f"{row['4. High Risk']:.2f}%"
        vhr = f"{row['5. Very High Risk']:.2f}%"
        
        html += f"""
        <tr style="border-bottom: 1px solid var(--secondary-background-color);">
            <td style="padding: 10px; font-weight: 600; color: var(--text-color);">{cohort}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{exc}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{gd}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{mod}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{hr}</td>
            <td style="padding: 10px; font-family: monospace; text-align:right; color: var(--text-color);">{vhr}</td>
        </tr>"""
    html += "</tbody></table>"
    return clean_html(html)

# Page Config
st.set_page_config(
    page_title="Credit Risk Scorecard & PD Platform",
    page_icon="https://img.icons8.com/external-flatart-icons-flat-flatarticons/64/external-bank-credit-and-loan-flatart-icons-flat-flatarticons.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and engine
@st.cache_resource
def get_scoring_engine():
    try:
        return ScoreGenerationEngine(
            scorecard_csv_path="outputs/scorecards/scorecard_table.csv",
            config_path="configs/config.yaml"
        )
    except Exception as e:
        st.error(f"Error loading scoring engine: {e}. Make sure the pipeline has been run.")
        return None

engine = get_scoring_engine()

# Detect current theme for conditional CSS injection
try:
    _theme_type = st.context.theme.type  # 'light', 'dark', or None
except Exception:
    _theme_type = None

is_dark = _theme_type == "dark"

# Custom Styling (Corporate Navy, Premium Fonts, and Harmonious Palettes)
st.markdown("""
<style>
    /* Custom Theme Variables for Light & Dark mode compatibility */
    :root {
        --nav-bg-active: #002B49; /* Corporate navy */
        --nav-text-active: #FFFFFF;
        --nav-border-active: #002B49;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --nav-bg-active: #0C3E66; /* Brightened corporate blue for contrast */
            --nav-text-active: #FFFFFF;
            --nav-border-active: #0C3E66;
        }
    }
    
    [data-theme="dark"] {
        --nav-bg-active: #0C3E66;
        --nav-text-active: #FFFFFF;
        --nav-border-active: #0C3E66;
    }
    
    [data-theme="light"] {
        --nav-bg-active: #002B49;
        --nav-text-active: #FFFFFF;
        --nav-border-active: #002B49;
    }

    /* Global Fonts & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
    }
    
    /* Sidebar background override */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid var(--secondary-background-color);
    }
    
    /* Custom Card Design */
    .premium-card {
        background-color: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.02);
        margin-bottom: 25px;
        color: var(--text-color);
    }
    
    /* KPI Cards In Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin-bottom: 25px;
    }
    .kpi-card {
        background-color: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: var(--text-color);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
        border-color: var(--primary-color);
    }
    
    /* Left border accent coloring for KPI cards */
    .kpi-loans { border-left: 5px solid #002B49; }
    .kpi-default { border-left: 5px solid #DC3545; }
    .kpi-auc { border-left: 5px solid #0284C7; }
    .kpi-ks { border-left: 5px solid #7C3AED; }
    .kpi-gini { border-left: 5px solid #D97706; }
    .kpi-psi { border-left: 5px solid #059669; }
    
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
        font-family: 'Outfit', sans-serif;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: var(--text-color);
        opacity: 0.6;
        margin-top: 4px;
    }
    
    /* Stepper (Project Journey) Styling */
    .stepper-container {
        padding-left: 20px;
        border-left: 2px solid var(--secondary-background-color);
        margin-left: 15px;
        margin-top: 15px;
    }
    .stepper-step {
        position: relative;
        padding-bottom: 25px;
    }
    .stepper-step::before {
        content: '';
        position: absolute;
        left: -29px;
        top: 4px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: var(--primary-color);
        border: 4px solid var(--background-color);
        box-shadow: 0 0 0 2px var(--primary-color);
    }
    .stepper-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 4px;
    }
    .stepper-desc {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.8;
        line-height: 1.5;
    }
    
    /* Badges for PSI/CSI Status */
    .badge-stable {
        color: #0F5132;
        background-color: #D1E7DD;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-drift {
        color: #842029;
        background-color: #F8D7DA;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-moderate {
        color: #664D03;
        background-color: #FFF3CD;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Variable contribution rows */
    .predictor-row {
        background-color: #F0FDF4;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 15px;
        border-left: 4px solid #16A34A;
        font-size: 0.85rem;
        color: #14532D;
    }
    .policy-row {
        background-color: var(--secondary-background-color);
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 15px;
        border-left: 4px solid #94A3B8;
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.9;
    }
    
    /* Custom decision cards */
    .decision-card {
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        color: white;
    }
    .decision-approved {
        background: linear-gradient(135deg, #0F5132 0%, #157347 100%);
        border-left: 6px solid #198754;
    }
    .decision-referred {
        background: linear-gradient(135deg, #664D03 0%, #8C6804 100%);
        border-left: 6px solid #FFC107;
    }
    .decision-declined {
        background: linear-gradient(135deg, #842029 0%, #A52834 100%);
        border-left: 6px solid #DC3545;
    }
    .decision-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        margin-bottom: 4px;
    }
    .decision-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        margin-bottom: 12px;
    }
    .decision-score {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 16px;
    }
    .decision-score span {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }
    .decision-metrics {
        display: flex;
        justify-content: space-between;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        padding-top: 12px;
        font-size: 0.95rem;
    }
    
    /* Styled Expander Header Card Overrides */
    div[data-testid="stExpander"] {
        background-color: var(--background-color) !important;
        border: 1px solid var(--secondary-background-color) !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stExpander"] details summary {
        background-color: var(--secondary-background-color) !important;
        padding: 12px 18px !important;
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        border-radius: 8px !important;
        transition: background-color 0.2s ease !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        background-color: var(--secondary-background-color) !important;
        opacity: 0.95;
    }
    
    /* ========================================
       SIDEBAR: FULLY NON-COLLAPSABLE SETUP
       ======================================== */
    
    /* Hide ALL collapse/expand control buttons in Streamlit */
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-header"],
    [data-testid="stSidebarCollapseButton"],
    .stSidebarCollapseButton,
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"],
    [data-testid="stSidebar"] > div:first-child > div > button,
    header button[kind="header"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    
    /* Keep sidebar always visible and at a fixed width */
    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
        width: 300px !important;
        transform: none !important;
        transition: none !important;
    }
    
    /* Prevent sidebar from going into collapsed state */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        display: block !important;
        visibility: visible !important;
        min-width: 300px !important;
    }
    
    /* PREMIUM SIDEBAR NAVIGATION OVERRIDES (LIGHT & DARK THEME COMPLIANT) */
    /* Hide the radio button circle inputs entirely */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    
    /* Override selector options wrapper gap */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0px !important;
    }
    
    /* Transform Streamlit options into clean vertical tabs */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        display: block !important;
        width: 100% !important;
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Individual category borders for inactive tabs */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1) {
        border-left: 4px solid #0D6EFD !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2) {
        border-left: 4px solid #198754 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3) {
        border-left: 4px solid #20C997 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(4) {
        border-left: 4px solid #6F42C1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(5) {
        border-left: 4px solid #D97706 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(6) {
        border-left: 4px solid #FD7E14 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(7) {
        border-left: 4px solid #DC3545 !important;
    }
    
    /* Selection hover effect */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        transform: translateX(4px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12) !important;
        filter: brightness(1.05);
    }
    
    /* Selection active state */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #002B49 0%, #004B7C 100%) !important;
        border-color: #002B49 !important;
        box-shadow: 0 4px 12px rgba(0, 43, 73, 0.35) !important;
    }
    
    /* Active indicators with neon highlights */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(1):has(input:checked) {
        border-left: 6px solid #00D2FF !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(2):has(input:checked) {
        border-left: 6px solid #2ECC71 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(3):has(input:checked) {
        border-left: 6px solid #1ABC9C !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(4):has(input:checked) {
        border-left: 6px solid #A55EEA !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(5):has(input:checked) {
        border-left: 6px solid #F1C40F !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(6):has(input:checked) {
        border-left: 6px solid #E67E22 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child(7):has(input:checked) {
        border-left: 6px solid #E74C3C !important;
    }
    
    /* White text on selected nav items */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Standard nav item text */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: var(--text-color) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        margin: 0 !important;
    }
    
   /* ========================================
   SIDEBAR TITLE
   ======================================== */

.sidebar-title-wrap {
    padding: 12px 8px 18px 8px;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    margin-bottom: 20px;
    margin-top: -20px;

    background: linear-gradient(
        145deg,
        #0F766E 0%,
        #2563EB 100%
    );

    border-radius: 10px;
}


.sidebar-title-main {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;

    color: white !important;

    letter-spacing: -0.02em;
}


.sidebar-title-sub {

    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;

    color: rgba(255,255,255,0.85) !important;

    text-transform: uppercase;
    letter-spacing: 0.06em;
}


/* Badge */

.sidebar-badge {

    display:inline-block;

    background: rgba(255,255,255,0.15);

    color:white !important;

    font-family:'Outfit',sans-serif;

    font-size:0.65rem;

    font-weight:700;

    letter-spacing:0.08em;

    padding:2px 8px;

    border-radius:20px;
}



/* Developer link */

.developer-link {

    text-decoration:none !important;

    color:#2563EB !important;

    font-weight:700 !important;
}


/* Streamlit dark mode compatibility */

.stApp[data-theme="dark"] .developer-link {

    color:#60A5FA !important;

}


/* Browser dark mode fallback */

@media(prefers-color-scheme:dark){

    .developer-link{

        color:#60A5FA !important;

    }

}
    /* ========================================
       APPLICANT SCORE PLACEMENT — vibrant title
       ======================================== */
    .score-placement-title {
        text-align: center;
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        margin: 15px 0 2px 0;
        padding: 6px 16px;
        display: inline-block;
        width: 100%;
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    [data-theme="dark"] .score-placement-title {
        background: linear-gradient(135deg, #00D2FF 0%, #7B61FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    @media (prefers-color-scheme: dark) {
        .score-placement-title {
            background: linear-gradient(135deg, #00D2FF 0%, #7B61FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
    }
    
    /* ========================================
       PSI/STABILITY BOX — readable in both modes
       ======================================== */
    .psi-box {
        padding: 15px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 5px solid #059669;
    }
    .psi-box-stable {
        background-color: rgba(5, 150, 105, 0.12);
    }
    .psi-rating {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 5px 0;
        color: #059669;
    }
    .psi-value {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        color: var(--text-color);
    }
    [data-theme="dark"] .psi-box-stable {
        background-color: rgba(5, 150, 105, 0.18);
    }
    [data-theme="dark"] .psi-rating {
        color: #34D399 !important;
    }
    @media (prefers-color-scheme: dark) {
        .psi-box-stable { background-color: rgba(5, 150, 105, 0.18); }
        .psi-rating { color: #34D399 !important; }
    }
    
    /* ========================================
       SVG ICONS SYSTEM
       ======================================== */
    .icon {
        width: 18px;
        height: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: middle;
        margin-right: 6px;
        color: var(--icon-color);
        stroke-width: 2px;
    }
    
    :root {
        --icon-color: #334155;
    }
    
    [data-theme="dark"] {
        --icon-color: #CBD5E1;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --icon-color: #CBD5E1;
        }
    }
    
    /* Bold text overrides */
    b, strong {
        font-weight: 700 !important;
    }

    /* ========================================
       LENDING POLICY SIMULATOR: KPI CARDS
       ======================================== */
    .sim-metric-card {
        background-color: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 12px;
        color: var(--text-color);
    }
    .sim-metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .sim-metric-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .sim-metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-color);
        font-family: 'Outfit', sans-serif;
        line-height: 1.2;
    }
    .sim-metric-sub {
        font-size: 0.72rem;
        color: var(--text-color);
        opacity: 0.55;
        margin-top: 4px;
    }
    
    .sim-approval { border-left: 4px solid #0D6EFD; }
    .sim-default { border-left: 4px solid #DC3545; }
    .sim-accounts { border-left: 4px solid #4F46E5; }
    .sim-volume { border-left: 4px solid #0891B2; }
    .sim-yield { border-left: 4px solid #D97706; }
    .sim-loss { border-left: 4px solid #EA580C; }
    .sim-netprofit { 
        border-left: 5px solid #16A34A;
        background: linear-gradient(135deg, var(--background-color) 0%, rgba(22, 163, 74, 0.08) 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Dynamic Icon and Primary Colors based on Streamlit Active Theme Type
if is_dark:
    st.markdown("""
    <style>
        :root {
            --icon-color: #CBD5E1 !important;
            --primary-color: #60A5FA !important;
            --color-blue: #60A5FA !important;
            --color-purple: #A78BFA !important;
            --color-teal: #2DD4BF !important;
            --color-amber: #FBBF24 !important;
            --color-green: #34D399 !important;
            --color-red: #F87171 !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        :root {
            --icon-color: #334155 !important;
            --primary-color: #2563EB !important;
            --color-blue: #2563EB !important;
            --color-purple: #7C3AED !important;
            --color-teal: #0F766E !important;
            --color-amber: #F59E0B !important;
            --color-green: #059669 !important;
            --color-red: #DC3545 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# JavaScript: Force sidebar to stay permanently expanded (non-collapsable)
st.markdown("""
<script>
(function keepSidebarExpanded() {
    function ensureExpanded() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.removeProperty('transform');
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
        }
        // Also hide any collapse buttons that may have appeared
        document.querySelectorAll(
            '[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], ' +
            'button[aria-label="Close sidebar"], button[aria-label="Open sidebar"]'
        ).forEach(el => { el.style.display = 'none'; });
    }
    // Run immediately and then observe DOM changes
    ensureExpanded();
    const observer = new MutationObserver(ensureExpanded);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-expanded', 'style'] });
})();
</script>
""", unsafe_allow_html=True)

# Sidebar Main Title
st.sidebar.markdown("""
<div class="sidebar-title-wrap">
    <div class="sidebar-title-main">Credit Risk Scorecard</div>
    <div class="sidebar-title-sub">&amp; Probability of Default</div>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigate Platform",
    [
        "Executive Overview",
        "Credit Risk Simulator",
        "Lending Policy Simulator",
        "Expected Credit Loss Framework",
        "Model Development Lifecycle",
        "Understanding Credit Risk Modeling",
        "Champion vs Challenger Performance",
        "Portfolio Stability & Monitoring"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style="margin-top:60px; border-top:1px solid rgba(128, 128, 128, 0.15); padding-top:20px; text-align:center;">
    <p style="color:var(--text-color); opacity:0.8; font-family:'Inter',sans-serif; font-size:0.78rem; margin:0;">
        Designed & Developed by <a href="https://www.linkedin.com/in/anam-sha/" target="_blank" class="developer-link">Anam</a>
    </p>
    <p style="color:var(--text-color); opacity:0.6; font-family:'Inter',sans-serif; font-size:0.72rem; margin:4px 0 0 0;">
        Risk Analytics | Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# The base CSS using var(--secondary-background-color) handles both light and dark modes:
# - Light mode: sidebar bg = #F0F2F6, nav buttons = #FFFFFF (secondary = white) → clear cards
# - Dark mode: sidebar bg = #0E1117, nav buttons = #262730 (secondary) → elevated dark cards


# ==========================================
# 1. Executive Overview
# ==========================================
if nav_selection == "Executive Overview":
    st.header("Executive Overview & Briefing")
    st.markdown("""
    <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; margin-top: -10px; margin-bottom: 25px; line-height: 1.5; opacity: 0.9; color: var(--text-color);">
        An interpretable machine learning framework for credit decisioning, risk assessment, and portfolio monitoring.
        <div style="font-size: 0.82rem; background-color: var(--secondary-background-color); padding: 6px 12px; border-radius: 4px; display: inline-block; border: 1px solid var(--secondary-background-color); margin-top: 10px;">
            Built by <b>Anam</b> &nbsp;|&nbsp; Machine Learning &bull; Credit Risk Analytics &bull; Business Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="premium-card">
        <h4 style="margin: 0 0 8px 0; font-weight:700;">Executive Summary</h4>
        <p style="margin: 0; font-size: 1rem; line-height: 1.6; opacity:0.9;">
            This platform demonstrates an end-to-end credit risk modeling framework that predicts probability of default and supports lending decisions using interpretable machine learning. Built utilizing historical retail lending portfolios, it establishes an automated underwriting mechanism that balances regulatory auditability requirements with advanced machine learning benchmarking.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom KPI Cards Grid with custom Left Border Color Accents
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card kpi-loans">
            <div class="kpi-title">Total Loans Analyzed</div>
            <div class="kpi-val">38,577</div>
            <div class="kpi-sub">In-Time Train: 18.0k &nbsp;|&nbsp; <b>Out-of-Time (2011)</b>: 20.5k</div>
        </div>
        <div class="kpi-card kpi-default">
            <div class="kpi-title">Default Rate</div>
            <div class="kpi-val">14.59%</div>
            <div class="kpi-sub"><b>Out-of-Time (2011)</b> rate: 15.87%</div>
        </div>
        <div class="kpi-card kpi-auc">
            <div class="kpi-title">ROC-AUC</div>
            <div class="kpi-val">0.6311</div>
            <div class="kpi-sub">Champion Scorecard <b>Out-of-Time (2011)</b></div>
        </div>
        <div class="kpi-card kpi-ks">
            <div class="kpi-title">KS Statistic</div>
            <div class="kpi-val">0.1870</div>
            <div class="kpi-sub">Good/Bad Separation</div>
        </div>
        <div class="kpi-card kpi-gini">
            <div class="kpi-title">Gini Coefficient</div>
            <div class="kpi-val">0.2622</div>
            <div class="kpi-sub">Calibration Measure</div>
        </div>
        <div class="kpi-card kpi-psi">
            <div class="kpi-title">PSI Stability Score</div>
            <div class="kpi-val" style="color:#059669;">0.0091</div>
            <div class="kpi-sub" style="color:#059669; font-weight:600;">Stable Population</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Key Architecture & Capabilities")
        st.markdown("""
        * **Decision Automation:** Real-time applicant scoring combining statistical scoring (points lookup) with policy checks.
        * **Explainable AI (XAI):** Mathematically transparent points contribution. Fully compliant with Fair Credit Reporting Act (FCRA) and ECOA audit standards.
        * **Champion-Challenger Testing:** Rigorous performance benchmarking of a Logistic Regression scorecard (Champion) against a non-linear XGBoost classifier (Challenger).
        * **Portfolio Stability Guardrails:** Population Stability Index (PSI) and Characteristic Stability Index (CSI) tracking to flag population drift.
        * **Business Policy Simulator:** Cutoff optimization leveraging expected credit loss (ECL), projected margins, and revenue trade-offs.
        """)
        
    with col2:
        st.subheader("Portfolio Risk Classification Tiers")
        if os.path.exists("outputs/reports/risk_band_distribution.png"):
            st.image("outputs/reports/risk_band_distribution.png", caption="Portfolio Risk Band Distribution", width="stretch")
        else:
            st.info("Risk band distribution plot not found.")


# ==========================================
# 2. Credit Risk Assessment Simulator
# ==========================================
elif nav_selection == "Credit Risk Simulator":
    st.header("Credit Risk Assessment Simulator")
    
    st.markdown("""
    <div style="background-color: var(--secondary-background-color); border-left: 5px solid var(--primary-color); padding: 15px; border-radius: 6px; margin-bottom: 20px; font-size: 0.95rem; line-height: 1.5; color: var(--text-color);">
        <b>Risk-Based Credit Decisioning:</b> Input the borrower profile below. 
        The system separates parameters into <b>Model Predictors</b> (which mathematically drive the risk score based on weight of evidence) and <b>Policy Info Only</b> (which are collected for policy bounds and structure checks but carry zero score weight).
    </div>
    """, unsafe_allow_html=True)
    
    if engine is None:
        st.warning("Production scoring engine is unavailable because scorecard CSV is missing.")
    else:
        col_calc1, col_calc2 = st.columns([3, 2])
        
        with col_calc1:
            st.subheader("Borrower Application Profile")
            
            with st.container():
                st.write("### Financial Profile")
                
                annual_inc = st.number_input(
                    "Annual Income ($):", 
                    min_value=0, max_value=5000000, value=65000, step=1000,
                    help="Model Predictor • Mathematically drives the risk score based on income bracket. Higher income reduces default risk."
                )
                
                dti = st.number_input(
                    "Debt-to-Income (DTI) Ratio (%):", 
                    min_value=0.0, max_value=100.0, value=15.2, step=0.1,
                    help="Policy Info Only: Debt-to-Income ratio. Used for structural checks. Does not carry score points."
                )
                
                home_ownership = st.selectbox(
                    "Home Ownership Status:", 
                    ["MORTGAGE", "RENT", "OWN", "OTHER"],
                    help="Policy Info Only: Home ownership status. Used for mortgage verification. Does not carry score points."
                )
                
            with st.container():
                st.write("### Credit Bureau History")
                
                loan_amnt = st.number_input(
                    "Requested Loan Amount ($):", 
                    min_value=500, max_value=100000, value=12000, step=500,
                    help="Policy Info Only: Requested loan volume. Collected for limit checks. Does not carry score points."
                )
                
                revol_util = st.number_input(
                    "Revolving Line Utilization (%):", 
                    min_value=0.0, max_value=150.0, value=45.0, step=1.0,
                    help="Model Predictor • Revolving credit line utilization percentage. High utilization increases credit risk."
                )
                
                credit_history_age = st.number_input(
                    "Credit History Age (Months):", 
                    min_value=0, max_value=1200, value=180, step=12,
                    help="Policy Info Only: Age of oldest credit line. Used for policy limits. Does not carry score points."
                )
                
                open_acc = st.number_input(
                    "Open Credit Accounts Count:", 
                    min_value=0, max_value=100, value=10, step=1,
                    help="Policy Info Only: Number of active open credit accounts. Used for exposure monitoring. Does not carry score points."
                )
                
            with st.container():
                st.write("### Derogatory History & Policy Rules")
                
                delinq_2yrs = st.number_input(
                    "Delinquencies (Past 24 Months):", 
                    min_value=0, max_value=50, value=0, step=1,
                    help="Policy Info Only: Number of 30+ days past-due occurrences. Used for delinquency policy bounds checks. Does not carry score points."
                )
                
                inq_last_6mths = st.number_input(
                    "Bureau Inquiries (Past 6 Months):", 
                    min_value=0, max_value=50, value=1, step=1,
                    help="Model Predictor • Number of hard credit inquiries in the last 6 months. High inquiry volume increases risk."
                )
                
                pub_rec = st.number_input(
                    "Derogatory Public Records:", 
                    min_value=0, max_value=50, value=0, step=1,
                    help="Model Predictor • Number of derogatory public records (tax liens, judgments). Public records increase risk."
                )
                
                pub_rec_bankruptcies = st.number_input(
                    "Public Record Bankruptcies Count:", 
                    min_value=0, max_value=50, value=0, step=1,
                    help="Model Predictor • Number of historical bankruptcy filings. Filings increase scorecard risk."
                )
                
                emp_length = st.slider(
                    "Employment Length (Years):", 
                    min_value=0, max_value=10, value=5, 
                    help="Policy Info Only: Employment stability indicator (years). Used for policy verification. Does not carry score points."
                )
                
                purpose = st.selectbox(
                    "Lending Loan Purpose:", 
                    ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car", "medical", "wedding", "other"],
                    help="Model Predictor • Primary intent of the loan. Segmented by risk log-odds severity."
                )
            
            applicant_record = {
                "annual_inc": annual_inc,
                "dti": dti,
                "home_ownership": home_ownership,
                "loan_amnt": loan_amnt,
                "revol_util": revol_util,
                "credit_history_age": credit_history_age,
                "open_acc": open_acc,
                "delinq_2yrs": delinq_2yrs,
                "inq_last_6mths": inq_last_6mths,
                "pub_rec": pub_rec,
                "pub_rec_bankruptcies": pub_rec_bankruptcies,
                "emp_length": emp_length,
                "purpose": purpose
            }
            
        with col_calc2:
            st.subheader("Decision Engine Output")
            
            # Run scoring engine
            res = engine.score_applicant(applicant_record)
            score = res['credit_score']
            pd_val = res['probability_of_default']
            risk_band = res['risk_band']
            decision = res['decision']
            
            # Choose class based on decision
            if decision == "Approved":
                decision_class = "decision-approved"
                rec_text = "Approve"
            elif decision == "Refer to Underwriting":
                decision_class = "decision-referred"
                rec_text = "Refer to Underwriting"
            else:
                decision_class = "decision-declined"
                rec_text = "Decline"
                
            st.markdown(f"""
            <div class="decision-card {decision_class}">
                <div class="decision-header">Automated Decision</div>
                <div class="decision-value">{rec_text}</div>
                <div class="decision-score">Credit Score: <span>{score}</span></div>
                <div class="decision-metrics">
                    <div><strong>Default Prob:</strong> {pd_val:.2%}</div>
                    <div><strong>Risk Category:</strong> {risk_band}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart title - vibrant gradient works in both modes
            st.markdown("<div class='score-placement-title'>Applicant Score Placement</div>", unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {
                        'range': [300, 850], 
                        'tickwidth': 1, 
                        'tickcolor': '#FAFAFA' if is_dark else '#31333F',
                        'tickfont': {'family': 'Inter', 'color': '#FAFAFA' if is_dark else '#31333F'}
                    },
                    'bar': {'color': '#0C3E66' if is_dark else '#002B49'},
                    'bgcolor': 'rgba(0,0,0,0.1)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [300, 600], 'color': 'rgba(239, 68, 68, 0.85)'},   # Vibrant Crimson Red
                        {'range': [600, 650], 'color': 'rgba(249, 115, 22, 0.85)'},  # Vibrant Orange
                        {'range': [650, 700], 'color': 'rgba(234, 179, 8, 0.85)'},   # Vibrant Yellow
                        {'range': [700, 750], 'color': 'rgba(74, 222, 128, 0.85)'},  # Vibrant Light Green
                        {'range': [750, 850], 'color': 'rgba(34, 197, 94, 0.85)'}   # Vibrant Green
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': '#FAFAFA' if is_dark else '#31333F', 'family': 'Outfit'},
                height=170,
                margin=dict(t=10, b=10, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, width="stretch")
            
            # Points Breakdown Table
            st.subheader("Scorecard Points Contribution")
            bkdown_df = pd.DataFrame([{
                "Predictor Characteristic": k.replace('_', ' ').title(),
                "Matched WoE Bin": res['bin_breakdown'].get(k, "Missing"),
                "Points Contribution": f"+{v}" if v >= 0 else str(v)
            } for k, v in res['score_breakdown'].items()])
            st.dataframe(bkdown_df, width="stretch", hide_index=True)


# ==========================================
# 3. Project Journey / Methodology
# ==========================================
elif nav_selection == "Model Development Lifecycle":
    st.header("Model Development Lifecycle Workflow")
    st.write("Below is the end-to-end model development lifecycle of this platform, showing how applicant data flows from source to policy execution.")
    
    st.markdown("""
    <div class="stepper-container">
        <div class="stepper-step">
            <div class="stepper-title">1. Data Ingestion & Target Definition</div>
            <div class="stepper-desc">
                Ingested 39,717 loan records from historical portfolios. Cleaned the default status: marked <b>Fully Paid</b> as Non-Defaults (0) and <b>Charged Off/Defaulted</b> as Defaults (1). Current active loans were removed to avoid classification bias.
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">2. Out-of-Time In-Time Segmentation</div>
            <div class="stepper-desc">
                Enforced temporal cohort segmentation: split data chronologically into <b>In-Time (2007-2010)</b> training sets and an <b>Out-of-Time (2011)</b> cohort (20,516 loans) to serve as validation representing real-world future cohorts.
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">3. Coarse Banning & Weight of Evidence (WoE) Mapping</div>
            <div class="stepper-desc">
                Binned raw variables using automated Decision Tree cuts to group similar risk segments. Computed <b>Weight of Evidence (WoE)</b> to convert categories into risk log-odds, ensuring monotonic default relationships.
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">4. Feature Selection via Information Value (IV)</div>
            <div class="stepper-desc">
                Calculated <b>Information Value (IV)</b> for all variables. Screened out predictive leakages and dropped indicators with useless predictive power (IV < 0.02), selecting 6 core predictors.
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">5. Logistic Scorecard Scaling</div>
            <div class="stepper-desc">
                Fit the selected predictors using a <b>Logistic Regression</b> champion model. Scaled odds mathematically using base score rules (Base Score of 600 at 50:1 odds, PDO of 20) to output integers suitable for banking systems.
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">6. XGBoost Challenger Benchmark</div>
            <div class="stepper-desc">
                Constructed a non-linear gradient-boosted tree model (XGBoost) on raw variables to serve as a challenger model, providing a raw performance baseline.
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">7. Model Validation & Verification</div>
            <div class="stepper-desc">
                Benchmarked models on the OOT cohort using validation tests: ROC-AUC curves, Kolmogorov-Smirnov (KS) separations, Gini coefficients, and probability calibration (Brier Score).
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">8. Stability & Drift Monitoring</div>
            <div class="stepper-desc">
                Quantified population changes between training and validation cohorts using <b>Population Stability Index (PSI)</b> and <b>Characteristic Stability Index (CSI)</b> to ensure stable risk parameters.
            </div>
        </div>
        <div class="stepper-step">
            <div class="stepper-title">9. Lending Policy Cutoff Simulation</div>
            <div class="stepper-desc">
                Simulated expected credit losses (ECL), approval volumes, interest yields, and net profits across various score cutoffs to determine the optimal business lending policy.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 4. Layman's Explainer -> Understanding Credit Risk Modeling
# ==========================================
elif nav_selection == "Understanding Credit Risk Modeling":
    st.header("Understanding Credit Risk Modeling")
    st.markdown("""
    <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px; line-height: 1.5; opacity: 0.9; color: var(--text-color);">
        A guided walkthrough of the core concepts, methodologies, and risk management principles that power modern lending decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Why Credit Risk Modeling Matters
    with st.expander("1. Why Credit Risk Modeling Matters", expanded=False):
        st.markdown(clean_html("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Why Credit Risk Modeling Matters</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            Banks lend capital upfront before knowing if a borrower will repay. The central challenge of retail lending is balancing 
            the need for credit growth, overall profitability, and risk control. A credit risk model acts as the steering wheel, 
            helping the bank estimate <strong>Probability of Default (PD)</strong>, project expected credit losses (ECL), and determine lending eligibility.
        </p>
        
        <div style="display: flex; justify-content: space-around; align-items: center; background-color: var(--secondary-background-color); padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid rgba(128,128,128,0.1);">
            <div style="text-align: center; flex: 1;">
                <div style="margin-bottom: 4px;">
                    <svg class="icon" style="color: var(--icon-color); width: 22px; height: 22px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                </div>
                <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-color); text-transform: uppercase; letter-spacing:0.02em; white-space: nowrap;">Loan Application</div>
            </div>
            <div style="padding: 0 10px;">
                <svg class="icon" style="width: 14px; height: 14px; opacity: 0.5; color: var(--icon-color);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="margin-bottom: 4px;">
                    <svg class="icon" style="color: var(--icon-color); width: 22px; height: 22px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                </div>
                <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-color); text-transform: uppercase; letter-spacing:0.02em; white-space: nowrap;">Risk Assessment</div>
            </div>
            <div style="padding: 0 10px;">
                <svg class="icon" style="width: 14px; height: 14px; opacity: 0.5; color: var(--icon-color);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="margin-bottom: 4px;">
                    <svg class="icon" style="color: var(--icon-color); width: 22px; height: 22px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                </div>
                <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-color); text-transform: uppercase; letter-spacing:0.02em; white-space: nowrap;">Prob. of Default</div>
            </div>
            <div style="padding: 0 10px;">
                <svg class="icon" style="width: 14px; height: 14px; opacity: 0.5; color: var(--icon-color);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="margin-bottom: 4px;">
                    <svg class="icon" style="color: var(--icon-color); width: 22px; height: 22px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                </div>
                <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-color); text-transform: uppercase; letter-spacing:0.02em; white-space: nowrap;">Credit Decision</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    
    # 2. What is Probability of Default (PD)?
    with st.expander("2. What is Probability of Default (PD)?"):
        st.markdown(textwrap.dedent("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Probability of Default (PD)</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            <strong>Probability of Default (PD)</strong> represents the mathematical likelihood that a borrower will fail to repay their debt obligation within a given timeframe (typically 12 months).
        </p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
            <div style="background: linear-gradient(135deg, rgba(22, 163, 74, 0.08) 0%, rgba(22, 163, 74, 0.01) 100%); border-left: 5px solid var(--color-green); padding: 12px 15px; border-radius: 8px;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <svg class="icon" style="color: var(--color-green); margin-right: 6px; width: 20px; height: 20px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                    <h5 style="margin: 0; color: var(--color-green); font-weight:700;">Applicant A (Low Risk)</h5>
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: var(--text-color); font-family: 'Outfit', sans-serif; margin-bottom: 4px;">PD = 2.0%</div>
                <p style="margin: 0; font-size: 0.82rem; opacity: 0.85; line-height: 1.4;">
                    High income, solid employment history, and low credit card utilization. Out-of-default expectation.
                </p>
            </div>
            <div style="background: linear-gradient(135deg, rgba(220, 53, 69, 0.08) 0%, rgba(220, 53, 69, 0.01) 100%); border-left: 5px solid var(--color-red); padding: 12px 15px; border-radius: 8px;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <svg class="icon" style="color: var(--color-red); margin-right: 6px; width: 20px; height: 20px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                    <h5 style="margin: 0; color: var(--color-red); font-weight:700;">Applicant B (High Risk)</h5>
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: var(--text-color); font-family: 'Outfit', sans-serif; margin-bottom: 4px;">PD = 18.0%</div>
                <p style="margin: 0; font-size: 0.82rem; opacity: 0.85; line-height: 1.4;">
                    Multiple recent credit bureau inquiries, high utilization, and derogatory public records. High default expectation.
                </p>
            </div>
        </div>
        
        <div style="font-size: 0.88rem; opacity: 0.85; line-height: 1.5; background-color: var(--secondary-background-color); padding: 8px 12px; border-radius: 6px; display: flex; align-items: center;">
            <svg class="icon" style="color: var(--color-blue); width: 18px; height: 18px; margin-right: 8px; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            <span><strong>Business Relevance:</strong> Lower PD allows banks to offer larger loan limits and lower interest rates. Higher PD requires either rejection, smaller limits, or charging a higher risk-based premium to cover potential losses.</span>
        </div>
        """), unsafe_allow_html=True)
        
    # 3. What is a Credit Scorecard?
    with st.expander("3. What is a Credit Scorecard?", expanded=False):
        st.markdown(textwrap.dedent("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="22"></line><line x1="5" y1="7" x2="19" y2="7"></line><path d="M5 7L2 17h6L5 7z"></path><path d="M19 7l-3 10h6l-3-10z"></path></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Credit Scorecard</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            A <strong>Credit Scorecard</strong> is a highly structured, points-based scoring system. It translates statistical model coefficients into simple lookup tables where points are assigned for each borrower characteristic.
        </p>
        
        <div style="display: flex; justify-content: space-around; align-items: center; background-color: var(--secondary-background-color); padding: 12px; border-radius: 8px; margin: 15px 0; border: 1px solid rgba(128,128,128,0.1); font-size: 0.8rem; font-weight:600;">
            <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                <svg class="icon" style="margin: 0; width: 22px; height: 22px; color: var(--icon-color);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                <span style="color: var(--text-color);">Characteristics</span>
            </div>
            <div>
                <svg class="icon" style="width: 18px; height: 18px; opacity: 0.8; color: var(--icon-color); margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </div>
            <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                <svg class="icon" style="margin: 0; width: 22px; height: 22px; color: var(--icon-color);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                <span style="color: var(--text-color);">Points Lookup</span>
            </div>
            <div>
                <svg class="icon" style="width: 18px; height: 18px; opacity: 0.8; color: var(--icon-color); margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </div>
            <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                <svg class="icon" style="margin: 0; width: 22px; height: 22px; color: var(--icon-color);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="9" x2="20" y2="9"></line><line x1="4" y1="15" x2="20" y2="15"></line><line x1="10" y1="3" x2="8" y2="21"></line><line x1="16" y1="3" x2="14" y2="21"></line></svg>
                <span style="color: var(--text-color);">Credit Score</span>
            </div>
            <div>
                <svg class="icon" style="width: 18px; height: 18px; opacity: 0.8; color: var(--icon-color); margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </div>
            <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                <svg class="icon" style="margin: 0; width: 22px; height: 22px; color: var(--icon-color);" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                <span style="color: var(--text-color);">Risk Tier</span>
            </div>
        </div>
        
        <h5 style="margin-top: 15px; margin-bottom: 6px; font-weight: 700;">Why Banks Prefer Scorecards:</h5>
        <ul style="font-size: 0.88rem; line-height: 1.5; opacity: 0.85; padding-left: 20px; margin: 0;">
            <li><strong>Transparency:</strong> Underwriters and customers can see exactly why a score was assigned.</li>
            <li><strong>Explainability:</strong> Direct alignment with fair lending laws (FCRA/ECOA) requiring clear reason codes for loan declines.</li>
            <li><strong>Auditable Logic:</strong> Easy for bank regulators to review, stress-test, and verify mathematical calibration.</li>
        </ul>
        """), unsafe_allow_html=True)
        
    # 4. Why Use Weight of Evidence (WoE)?
    with st.expander("4. Why Use Weight of Evidence (WoE)?", expanded=False):
        st.markdown(textwrap.dedent("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Weight of Evidence (WoE) Transformation</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            <strong>Weight of Evidence (WoE)</strong> is a data transformation technique. Instead of using raw inputs directly (e.g. continuous utilization % or income numbers), continuous characteristics are grouped into risk-based segments.
        </p>
        
        <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); border-radius: 8px; padding: 15px; margin: 12px 0;">
            <h5 style="margin-top:0; margin-bottom: 8px; font-weight: 700; font-size: 0.88rem;">Example: Income Categorization</h5>
            <div style="display: flex; gap: 12px; align-items: stretch; justify-content: center; font-size: 0.8rem; text-align: center;">
                <div style="flex: 1; background-color: var(--background-color); padding: 10px; border-radius: 6px; display: flex; flex-direction: column; align-items: center;">
                    <span style="font-weight:700; display:block; margin-bottom:4px; color: var(--text-color);">1. Income Bands</span>
                    <div style="display: flex; align-items: center; gap: 4px; margin-top: 4px; color: var(--text-color);">
                        <svg class="icon" style="width: 16px; height: 16px; color: var(--icon-color); margin: 0 4px 0 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                        <span>Low (&lt;$30k)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 4px; margin-top: 4px; color: var(--text-color);">
                        <svg class="icon" style="width: 16px; height: 16px; color: var(--icon-color); margin: 0 4px 0 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                        <span>Medium ($30k-$75k)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 4px; margin-top: 4px; color: var(--text-color);">
                        <svg class="icon" style="width: 16px; height: 16px; color: var(--icon-color); margin: 0 4px 0 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                        <span>High (&gt;$75k)</span>
                    </div>
                </div>
                <div style="align-self: center; opacity: 0.5;">
                    <svg class="icon" style="width: 18px; height: 18px; color: var(--icon-color); opacity: 0.8; margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </div>
                <div style="flex: 1; background-color: var(--background-color); padding: 10px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <span style="font-weight:700; display:block; margin-bottom:4px; color: var(--text-color);">2. WoE Conversion</span>
                    <div style="display: flex; align-items: center; gap: 4px; margin-top: 4px; color: var(--text-color);">
                        <svg class="icon" style="color: #DC3545; width: 14px; height: 14px; margin: 0 4px 0 0; fill: currentColor;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle></svg>
                        <span>Negative (High Risk)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 4px; margin-top: 4px; color: var(--text-color);">
                        <svg class="icon" style="color: #D97706; width: 14px; height: 14px; margin: 0 4px 0 0; fill: currentColor;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle></svg>
                        <span>Neutral (Avg Risk)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 4px; margin-top: 4px; color: var(--text-color);">
                        <svg class="icon" style="color: #198754; width: 14px; height: 14px; margin: 0 4px 0 0; fill: currentColor;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle></svg>
                        <span>Positive (Low Risk)</span>
                    </div>
                </div>
                <div style="align-self: center; opacity: 0.5;">
                    <svg class="icon" style="width: 18px; height: 18px; color: var(--icon-color); opacity: 0.8; margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </div>
                <div style="flex: 1; background-color: var(--background-color); padding: 10px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <span style="font-weight:700; display:block; margin-bottom:4px; color: var(--text-color);">3. Scorecard Points</span>
                    <span style="color: #DC3545; font-weight: 600; margin-top: 4px;">-15 points</span>
                    <span style="color: #D97706; font-weight: 600; margin-top: 4px;">+10 points</span>
                    <span style="color: #198754; font-weight: 600; margin-top: 4px;">+45 points</span>
                </div>
            </div>
        </div>
        
        <h5 style="margin-top: 15px; margin-bottom: 6px; font-weight: 700;">Key Operational Advantages:</h5>
        <ul style="font-size: 0.88rem; line-height: 1.5; opacity: 0.85; padding-left: 20px; margin: 0;">
            <li><strong>Robust to Outliers:</strong> Extreme values (e.g. extremely high or zero income) are kept in stable bins, protecting model reliability.</li>
            <li><strong>Linearizes Risk:</strong> WoE maps non-linear raw variables into a linear relationship with log-odds, making standard logistic models highly accurate.</li>
            <li><strong>Regulatory Acceptance:</strong> Establishes explainable, monotonic risk patterns required by banking supervision committees.</li>
        </ul>
        """), unsafe_allow_html=True)
        
    # 5. What is Information Value (IV)?
    with st.expander("5. What is Information Value (IV)?", expanded=False):
        st.markdown(textwrap.dedent("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Information Value (IV)</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            <strong>Information Value (IV)</strong> is a statistical metric that measures the predictive power of a single characteristic in separating defaults from non-defaults.
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; text-align: center;">
            <div style="background-color: var(--secondary-background-color); padding: 8px; border-radius: 6px; border-top: 4px solid #6C757D;">
                <strong style="display:block; font-size: 0.78rem; color: #6C757D;">&lt; 0.02</strong>
                <span style="font-size: 0.72rem; font-weight:700; color: var(--text-color);">Not Predictive</span>
            </div>
            <div style="background-color: var(--secondary-background-color); padding: 8px; border-radius: 6px; border-top: 4px solid #FFC107;">
                <strong style="display:block; font-size: 0.78rem; color: #FFC107;">0.02 - 0.10</strong>
                <span style="font-size: 0.72rem; font-weight:700; color: var(--text-color);">Weak</span>
            </div>
            <div style="background-color: var(--secondary-background-color); padding: 8px; border-radius: 6px; border-top: 4px solid #0D6EFD;">
                <strong style="display:block; font-size: 0.78rem; color: #0D6EFD;">0.10 - 0.30</strong>
                <span style="font-size: 0.72rem; font-weight:700; color: var(--text-color);">Strong</span>
            </div>
            <div style="background-color: var(--secondary-background-color); padding: 8px; border-radius: 6px; border-top: 4px solid #198754;">
                <strong style="display:block; font-size: 0.78rem; color: #198754;">&gt; 0.30</strong>
                <span style="font-size: 0.72rem; font-weight:700; color: var(--text-color);">Very Strong</span>
            </div>
        </div>
        
        <div style="font-size: 0.88rem; opacity: 0.85; line-height: 1.5; background-color: var(--secondary-background-color); padding: 8px 12px; border-radius: 6px; display: flex; align-items: center;">
            <svg class="icon" style="color: var(--color-blue); width: 18px; height: 18px; margin-right: 8px; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            <span><strong>Why it is used:</strong> IV acts as an initial filter. Weak predictors are dropped from scorecard consideration to prevent model clutter and minimize underwriting data request costs.</span>
        </div>
        """), unsafe_allow_html=True)
        
    # 6. Champion vs Challenger Models
    with st.expander("6. Champion vs Challenger Models", expanded=False):
        st.markdown(clean_html("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"></circle><circle cx="6" cy="6" r="3"></circle><path d="M13 6h3a2 2 0 0 1 2 2v7"></path><line x1="6" y1="9" x2="6" y2="21"></line></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Model Benchmarking: Champion vs Challenger</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            Before deploying any model into production, banks compare their currently active model (the <strong>Champion</strong>) against a newly developed model (the <strong>Challenger</strong>) to analyze performance trade-offs.
        </p>
        
        <div style="display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 20px; margin-top: 15px; margin-bottom: 15px;">
            <!-- Champion Card -->
            <div style="background-color: var(--background-color); padding: 20px; border-radius: 10px; border: 2px solid var(--color-blue); box-shadow: 0 4px 12px rgba(13, 110, 253, 0.08); position: relative;">
                <span style="position: absolute; top: 15px; right: 15px; background-color: rgba(13, 110, 253, 0.15); color: var(--color-blue); font-size: 0.68rem; font-weight: 700; padding: 3px 8px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid rgba(13, 110, 253, 0.25);">Active Production</span>
                
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <svg class="icon" style="color: var(--color-blue); width: 24px; height: 24px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 11 11 13 15 9"></polyline></svg>
                    <h5 style="margin: 0; font-size: 1.05rem; font-weight: 800; color: var(--text-color);">Champion Scorecard</h5>
                </div>
                
                <div style="font-size: 0.85rem; margin-bottom: 12px; color: var(--text-color); opacity: 0.9;">
                    <div style="margin-bottom: 6px;"><strong>Model Type:</strong> Logistic Regression (Weight of Evidence)</div>
                    <div style="margin-bottom: 6px;"><strong>Strengths:</strong> Mathematically transparent, fully explainable log-odds contribution, regulatory compliant, stable probability calibration.</div>
                    <div style="margin-bottom: 6px;"><strong>Limitations:</strong> Assumes linear variable relationships, misses complex non-linear interaction effects unless manually engineered.</div>
                    <div><strong>Typical Banking Usage:</strong> Core credit underwriting decisions, interest rate risk-based pricing, regulatory capital calculations.</div>
                </div>
            </div>
            
            <!-- Challenger Card -->
            <div style="background-color: var(--secondary-background-color); padding: 20px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.2); opacity: 0.9;">
                <span style="float: right; background-color: var(--secondary-background-color); color: var(--text-color); font-size: 0.68rem; font-weight: 700; padding: 3px 8px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid rgba(128,128,128,0.2); opacity: 0.6;">Challenger benchmark</span>
                
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <svg class="icon" style="color: var(--color-purple); width: 24px; height: 24px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"></circle><circle cx="6" cy="6" r="3"></circle><path d="M13 6h3a2 2 0 0 1 2 2v7"></path><line x1="6" y1="9" x2="6" y2="21"></line></svg>
                    <h5 style="margin: 0; font-size: 1.05rem; font-weight: 800; color: var(--text-color);">Challenger Model</h5>
                </div>
                
                <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.95;">
                    <div style="margin-bottom: 6px;"><strong>Model Type:</strong> XGBoost (Gradient Boosted Trees)</div>
                    <div style="margin-bottom: 6px;"><strong>Strengths:</strong> High predictive accuracy, automatically detects multi-variable interaction effects and non-linear patterns.</div>
                    <div style="margin-bottom: 6px;"><strong>Limitations:</strong> "Black box" nature requires surrogate model explainers, prone to poor probability calibration (Brier Score).</div>
                    <div><strong>Typical Banking Usage:</strong> Fraud detection systems, credit pre-screening marketing campaigns, cross-sell recommendation engines.</div>
                </div>
            </div>
        </div>
        
        <div style="font-size: 0.88rem; opacity: 0.85; line-height: 1.5; background-color: var(--secondary-background-color); padding: 8px 12px; border-radius: 6px; display: flex; align-items: center;">
            <svg class="icon" style="color: var(--color-blue); width: 18px; height: 18px; margin-right: 8px; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            <span><strong>Industrial Standard:</strong> Banks select the model that provides the best balance of raw predictive power, calibration stability, operational execution costs, and regulatory compliance.</span>
        </div>
        """), unsafe_allow_html=True)
        
    # 7. Why Validate Models?
    with st.expander("7. Why Validate Models?", expanded=False):
        st.markdown(textwrap.dedent("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Model Validation Metrics</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            A credit risk model must maintain performance on unseen future borrowers. Validation processes test the model across multiple statistical dimensions:
        </p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 5px;">
            <div style="background-color: var(--secondary-background-color); padding: 10px 15px; border-radius: 6px; display: flex; align-items: flex-start; gap: 10px;">
                <svg class="icon" style="color: var(--color-blue); width: 24px; height: 24px; margin: 0; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                <div>
                    <strong style="font-size:0.85rem; color:var(--text-color);">ROC-AUC (Area Under the Curve)</strong>
                    <p style="margin: 3px 0 0 0; font-size: 0.78rem; opacity: 0.85; line-height:1.4;">
                        Evaluates discrimination power—how well the model separates good borrowers from defaults.
                    </p>
                </div>
            </div>
            <div style="background-color: var(--secondary-background-color); padding: 10px 15px; border-radius: 6px; display: flex; align-items: flex-start; gap: 10px;">
                <svg class="icon" style="color: var(--color-purple); width: 24px; height: 24px; margin: 0; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                <div>
                    <strong style="font-size:0.85rem; color:var(--text-color);">Kolmogorov-Smirnov (KS)</strong>
                    <p style="margin: 3px 0 0 0; font-size: 0.78rem; opacity: 0.85; line-height:1.4;">
                        Measures the maximum separation point between the cumulative score distributions of good and bad borrowers.
                    </p>
                </div>
            </div>
            <div style="background-color: var(--secondary-background-color); padding: 10px 15px; border-radius: 6px; display: flex; align-items: flex-start; gap: 10px;">
                <svg class="icon" style="color: var(--color-teal); width: 24px; height: 24px; margin: 0; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>
                <div>
                    <strong style="font-size:0.85rem; color:var(--text-color);">Gini Coefficient</strong>
                    <p style="margin: 3px 0 0 0; font-size: 0.78rem; opacity: 0.85; line-height:1.4;">
                        A normalized metric showing default concentration across score deciles (Gini = 2 * AUC - 1).
                    </p>
                </div>
            </div>
            <div style="background-color: var(--secondary-background-color); padding: 10px 15px; border-radius: 6px; display: flex; align-items: flex-start; gap: 10px;">
                <svg class="icon" style="color: var(--color-amber); width: 24px; height: 24px; margin: 0; flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
                <div>
                    <strong style="font-size:0.85rem; color:var(--text-color);">Brier Score (Calibration)</strong>
                    <p style="margin: 3px 0 0 0; font-size: 0.78rem; opacity: 0.85; line-height:1.4;">
                        Quantifies the accuracy of predicted probability values compared to actual default frequencies.
                    </p>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
    # 8. Why Monitor Models After Deployment?
    with st.expander("8. Why Monitor Models After Deployment?", expanded=False):
        st.markdown(textwrap.dedent("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Post-Deployment Monitoring</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 12px;">
            A deployed credit scorecard is vulnerable to <strong>Population Drift</strong> due to shifting economic cycles, credit policy adjustments, or changing demographics.
        </p>
        
        <div style="background-color: var(--secondary-background-color); border-radius: 8px; border-left: 5px solid var(--color-amber); padding: 12px 15px; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 6px; gap: 6px;">
                <svg class="icon" style="color: var(--color-amber); width: 20px; height: 20px; margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                <h5 style="margin: 0; color: var(--color-amber); font-weight:700; font-size: 0.85rem;">Drift Concept Example:</h5>
            </div>
            <div style="display: flex; gap: 12px; align-items: center; justify-content: center; font-size: 0.78rem; text-align: center;">
                <div style="flex: 1; background-color: var(--background-color); padding: 6px; border-radius: 4px; display: flex; align-items: center; justify-content: center; gap: 6px;">
                    <svg class="icon" style="width: 18px; height: 18px; color: var(--icon-color); margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>
                    <span style="color: var(--text-color);"><strong>Training Cohort:</strong> Salaried employees</span>
                </div>
                <div style="opacity: 0.5;">
                    <svg class="icon" style="width: 18px; height: 18px; color: var(--icon-color); opacity: 0.8; margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </div>
                <div style="flex: 1; background-color: var(--background-color); padding: 6px; border-radius: 4px; display: flex; align-items: center; justify-content: center; gap: 6px;">
                    <svg class="icon" style="width: 18px; height: 18px; color: var(--icon-color); margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                    <span style="color: var(--text-color);"><strong>Future Applications:</strong> Gig-economy workers</span>
                </div>
                <div style="opacity: 0.5;">
                    <svg class="icon" style="width: 18px; height: 18px; color: var(--icon-color); opacity: 0.8; margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </div>
                <div style="flex: 1; background: rgba(220, 53, 69, 0.08); padding: 6px; border-radius: 4px; border: 1px dashed rgba(220, 53, 69, 0.2); display: flex; align-items: center; justify-content: center; gap: 6px;">
                    <svg class="icon" style="color: var(--color-red); width: 18px; height: 18px; margin: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                    <span style="color: var(--color-red); font-weight:600;">Drift Detected</span>
                </div>
            </div>
        </div>
        
        <h5 style="margin-top: 15px; margin-bottom: 6px; font-weight: 700;">Drift Metrics:</h5>
        <ul style="font-size: 0.88rem; line-height: 1.5; opacity: 0.85; padding-left: 20px; margin: 0;">
            <li><strong>Population Stability Index (PSI):</strong> Quantifies overall shift in score distribution between training and production cohorts.</li>
            <li><strong>Characteristic Stability Index (CSI):</strong> Pinpoints which specific applicant characteristics (e.g. recent inquiries) are causing the drift.</li>
        </ul>
        """), unsafe_allow_html=True)
        
    # 9. End-to-End Lending Journey Summary
    with st.expander("9. End-to-End Lending Journey Lifecycle", expanded=False):
        st.markdown(textwrap.dedent("""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <svg class="icon" style="color: var(--color-blue); width: 22px; height: 22px; margin-right: 8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
            <h5 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--text-color);">Lending Risk Operations Sequence</h5>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; opacity: 0.9; margin-bottom: 20px;">
            Below is the full sequence of risk operations that govern credit decisioning:
        </p>
        
        <div style="display: flex; flex-direction: column; gap: 15px; margin-left: 20px; border-left: 2px solid var(--primary-color); padding-left: 25px;">
            <div style="position: relative;">
                <div style="position: absolute; left: -33px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background-color: var(--primary-color);"></div>
                <strong style="color: var(--text-color); font-size: 0.9rem;">1. Data Collection & Pre-processing</strong>
                <p style="margin: 2px 0 0 0; font-size: 0.8rem; opacity: 0.85;">Aggregating credit files; handling missing data; splitting populations chronologically into in-time training and out-of-time validation cohorts.</p>
            </div>
            <div style="position: relative;">
                <div style="position: absolute; left: -33px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background-color: var(--primary-color);"></div>
                <strong style="color: var(--text-color); font-size: 0.9rem;">2. Binning & WoE Conversion</strong>
                <p style="margin: 2px 0 0 0; font-size: 0.8rem; opacity: 0.85;">Converting continuous characteristics into risk categories based on monotonic default probability trends.</p>
            </div>
            <div style="position: relative;">
                <div style="position: absolute; left: -33px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background-color: var(--primary-color);"></div>
                <strong style="color: var(--text-color); font-size: 0.9rem;">3. Scorecard Modeling & Validation</strong>
                <p style="margin: 2px 0 0 0; font-size: 0.8rem; opacity: 0.85;">Fitting a logistic model; scaling points; performing Champion-Challenger validation checks on validation datasets.</p>
            </div>
            <div style="position: relative;">
                <div style="position: absolute; left: -33px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background-color: var(--primary-color);"></div>
                <strong style="color: var(--text-color); font-size: 0.9rem;">4. Underwriting & Policy Cutoffs</strong>
                <p style="margin: 2px 0 0 0; font-size: 0.8rem; opacity: 0.85;">Simulating yields and losses to pick the optimal score cutoff threshold that satisfies the bank's risk appetite constraints.</p>
            </div>
            <div style="position: relative;">
                <div style="position: absolute; left: -33px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background-color: var(--primary-color);"></div>
                <strong style="color: var(--text-color); font-size: 0.9rem;">5. Production Execution & Monitoring</strong>
                <p style="margin: 2px 0 0 0; font-size: 0.8rem; opacity: 0.85;">Deciding on live applicant requests in real-time; monitoring daily pipelines using PSI/CSI alerts to catch population shifts.</p>
            </div>
        </div>
        """), unsafe_allow_html=True)



# ==========================================
# 5. Model Performance Dashboard
# ==========================================
elif nav_selection == "Champion vs Challenger Performance":
    st.header("Champion vs Challenger Model Performance")
    
    st.markdown("""
    <div style="background-color: var(--secondary-background-color); border-left: 5px solid var(--primary-color); padding: 15px; border-radius: 6px; margin-bottom: 20px; font-size: 0.95rem; color: var(--text-color);">
        <b>Champion Selection Note:</b> The bank retains the <b>Logistic Scorecard</b> as the select Champion. 
        While the <b>XGBoost Challenger</b> demonstrates a marginal lift in separation (ROC-AUC of 0.6331 vs 0.6311), 
        the Scorecard achieves <b>superior probability calibration</b> (Brier Score of 0.1308 vs 0.2274), 
        ensuring that predicted default probabilities align closely with actual defaults for pricing, alongside complete compliance compliance.
    </div>
    """, unsafe_allow_html=True)
    
    # Load champion-challenger report
    report_path = "outputs/monitoring/champion_challenger_report.xlsx"
    if os.path.exists(report_path):
        comp_df = pd.read_excel(report_path)
        st.markdown(make_cc_table_html(comp_df), unsafe_allow_html=True)
    else:
        st.warning("Performance report Excel file not found.")
        
    st.subheader("Out-of-Time (2011) Performance Curves")
    col_curve1, col_curve2 = st.columns(2)
    
    with col_curve1:
        if os.path.exists("outputs/reports/roc_curve.png"):
            st.image("outputs/reports/roc_curve.png", caption="ROC Curve (Out-of-Time (2011) cohort validation)", width="stretch")
        else:
            st.info("ROC plot image not found.")
            
    with col_curve2:
        if os.path.exists("outputs/reports/ks_curve.png"):
            st.image("outputs/reports/ks_curve.png", caption="Kolmogorov-Smirnov Separation Curve", width="stretch")
        else:
            st.info("KS plot image not found.")
            
    st.subheader("Model Validation & Calibration")
    col_curve3, col_curve4 = st.columns([1, 1])
    with col_curve3:
        if os.path.exists("outputs/reports/calibration_curve.png"):
            st.image("outputs/reports/calibration_curve.png", caption="Model Probability Calibration Curve", width="stretch")
        else:
            st.info("Calibration curve plot not found.")
            
    with col_curve4:
        st.markdown("""
        #### Understanding Calibration in Lending
        Probability calibration is crucial for financial institutions. If a model predicts an applicant has a 10% Probability of Default, then exactly 10 out of 100 such applicants should default.
        
        * **Logistic Regression (Champion):** Demonstrates close alignment with the diagonal 45-degree calibration line (lower Brier Score of 0.1308). This ensures reliable risk-based pricing and reserve calculation metrics.
        * **XGBoost (Challenger):** Displays over-optimistic or under-optimistic probabilities in intermediate bins (high Brier Score of 0.2274), which could lead to mispriced loans and incorrect Expected Loss reserves.
        """)


# ==========================================
# 6. Risk Monitoring Dashboard
# ==========================================
elif nav_selection == "Portfolio Stability & Monitoring":
    st.header("Risk Stability & Monitoring")
    st.markdown("Validation metrics to detect risk drift and score migrations between In-Time training cohorts and the **Out-of-Time (2011)** validation cohort.", unsafe_allow_html=True)
    
    col_mon1, col_mon2 = st.columns([1, 1])
    
    with col_mon1:
        st.subheader("Population Stability Index (PSI)")
        psi_xls_path = "outputs/monitoring/psi_report.xlsx"
        if os.path.exists(psi_xls_path):
            psi_sum = pd.read_excel(psi_xls_path, sheet_name='PSI Summary')
            psi_details = pd.read_excel(psi_xls_path, sheet_name='PSI Details')
            
            psi_val = psi_sum['Value'].values[0]
            psi_interp = psi_sum['Interpretation'].values[0]
            
            # Shield check icon instead of ✅ emoji
            st.markdown(f"""
            <div class="psi-box psi-box-stable">
                <div class="psi-rating" style="display: flex; align-items: center; gap: 6px;">
                    <svg class="icon" style="color: #059669; width: 18px; height: 18px; margin: 0; flex-shrink:0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 11 11 13 15 9"></polyline></svg>
                    <span>Stability Rating: {psi_interp}</span>
                </div>
                <div class="psi-value">Total Portfolio PSI: <b>{psi_val:.5f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(make_psi_details_html(psi_details), unsafe_allow_html=True)
        else:
            st.info("PSI stability report not found.")
            
    with col_mon2:
        st.subheader(
            "Characteristic Stability Index (CSI)",
            help="Drift Alert: pub_rec_bankruptcies has crossed the significant drift threshold (CSI = 0.2860). This is likely caused by the shift in the bankruptcies data distribution in the Out-of-Time (2011) validation cohort. Underwriting policy rules should adjust bankruptcies limits to mitigate drift."
        )
        csi_xls_path = "outputs/monitoring/csi_report.xlsx"
        if os.path.exists(csi_xls_path):
            csi_sum = pd.read_excel(csi_xls_path, sheet_name='CSI Summary')
            st.markdown(make_csi_table_html(csi_sum), unsafe_allow_html=True)
        else:
            st.info("CSI stability report not found.")
            
    # Score distribution visuals
    st.subheader("Score Migration & Distribution Shift")
    col_mon_img1, col_mon_img2 = st.columns([1, 1])
    with col_mon_img1:
        if os.path.exists("outputs/reports/score_distribution.png"):
            st.image("outputs/reports/score_distribution.png", caption="Score Shift: In-Time Train vs Out-of-Time (2011) Cohort", width="stretch")
        else:
            st.info("Score distribution comparison image not found.")
            
    with col_mon_img2:
        score_mon_xls = "outputs/monitoring/score_monitoring_report.xlsx"
        if os.path.exists(score_mon_xls):
            mig_df = pd.read_excel(score_mon_xls, sheet_name='Quarterly Migration Matrix', index_col=0)
            st.markdown("Quarterly Score Migration matrix (**Out-of-Time (2011)** Validation):")
            st.markdown(make_mig_table_html(mig_df), unsafe_allow_html=True)
        else:
            st.info("Score monitoring report not found.")


# ==========================================
# 7. Lending Policy Simulator
# ==========================================
elif nav_selection == "Lending Policy Simulator":
    st.header("Lending Policy Simulation & Cutoff Optimizer")
    st.markdown("""
    <div style="background-color: var(--secondary-background-color); border-left: 5px solid var(--primary-color); padding: 15px; border-radius: 6px; margin-bottom: 20px; font-size: 0.95rem; color: var(--text-color);">
        <b>Lending Strategy Simulator:</b> Adjust the score cutoff threshold below. The engine dynamically calculates estimated approval rates, portfolio default risk, and net margin outcomes based on financial metrics.
    </div>
    """, unsafe_allow_html=True)
    
    sim_path = "outputs/reports/approval_rate_analysis.csv"
    if os.path.exists(sim_path):
        sim_df = pd.read_csv(sim_path)
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            cutoff_val = st.slider(
                "Select Score Cutoff Threshold:", 
                min_value=int(sim_df['cutoff'].min()), 
                max_value=int(sim_df['cutoff'].max()), 
                value=620, 
                step=10
            )
            
            # Retrieve simulation data for selected cutoff
            cutoff_row = sim_df[sim_df['cutoff'] == cutoff_val].iloc[0]
            
            st.markdown("### Portfolio Metrics")
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px;">
                <div class="sim-metric-card sim-approval">
                    <div class="sim-metric-title">Approval Rate</div>
                    <div class="sim-metric-val">{cutoff_row['approval_rate_pct']:.1f}%</div>
                    <div class="sim-metric-sub">Target Underwriting</div>
                </div>
                <div class="sim-metric-card sim-default">
                    <div class="sim-metric-title">Default Rate</div>
                    <div class="sim-metric-val">{cutoff_row['actual_bad_rate_pct']:.2f}%</div>
                    <div class="sim-metric-sub">Expected Bad Rate</div>
                </div>
                <div class="sim-metric-card sim-accounts">
                    <div class="sim-metric-title">Approved Loans</div>
                    <div class="sim-metric-val">{int(cutoff_row['approved_loans']):,}</div>
                    <div class="sim-metric-sub">OOT Accounts</div>
                </div>
                <div class="sim-metric-card sim-volume">
                    <div class="sim-metric-title">Approved Vol</div>
                    <div class="sim-metric-val">${cutoff_row['approved_amount']/1e6:.2f}M</div>
                    <div class="sim-metric-sub">Exposure</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Financial Projections")
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 12px;">
                <div class="sim-metric-card sim-yield">
                    <div class="sim-metric-title">Interest Yield</div>
                    <div class="sim-metric-val">${cutoff_row['expected_revenue']/1e6:.2f}M</div>
                    <div class="sim-metric-sub">Gross Revenue</div>
                </div>
                <div class="sim-metric-card sim-loss">
                    <div class="sim-metric-title">Credit Loss (ECL)</div>
                    <div class="sim-metric-val">${cutoff_row['expected_loss']/1e6:.2f}M</div>
                    <div class="sim-metric-sub">Expected Loss</div>
                </div>
            </div>
            <div class="sim-metric-card sim-netprofit" style="margin-top: 5px; margin-bottom: 20px;">
                <div class="sim-metric-title">Expected Net Profit Yield</div>
                <div class="sim-metric-val" style="font-size: 1.8rem; color: #22C55E;">${cutoff_row['net_profit']/1e6:.2f}M</div>
                <div class="sim-metric-sub">Portfolio Net Margin</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="font-size: 0.75rem; opacity: 0.6; line-height: 1.4; border-top: 1px solid rgba(128,128,128,0.15); padding-top: 10px; margin-top: 15px;">
                *Note: Parameters are scaled using interest calculations from config.yaml (Avg Interest Rate: 12.0%, LGD: 60%, Term: 3 Years).*
            </div>
            """, unsafe_allow_html=True)
            
        with col_sim2:
            st.subheader("Trade-off & Profit Curves")
            # Interactive Plotly Tradeoff
            fig_trade = go.Figure()
            fig_trade.add_trace(go.Scatter(
                x=sim_df['cutoff'], y=sim_df['approval_rate_pct'],
                mode='lines', name='Approval Rate (%)', line=dict(color='#2563EB', width=2.5)
            ))
            fig_trade.add_trace(go.Scatter(
                x=sim_df['cutoff'], y=sim_df['actual_bad_rate_pct'],
                mode='lines', name='Expected Bad Rate (%)', line=dict(color='#DC3545', width=2, dash='dash'),
                yaxis='y2'
            ))
            
            fig_trade.update_layout(
                title="Lending Policy Tradeoff: Approval vs Bad Rate",
                xaxis=dict(title='Score Cutoff'),
                yaxis=dict(title=dict(text='Approval Rate (%)', ), tickfont=dict(color='#002B49')),
                yaxis2=dict(title=dict(text='Bad Rate (%)', font=dict(color='#DC3545')), tickfont=dict(color='#DC3545'), anchor='x', overlaying='y', side='right'),
                legend=dict(x=0.02, y=0.98),
                height=350
            )
            st.plotly_chart(fig_trade, width="stretch")
            
            # Interactive Profit Curve
            fig_prof = px.line(
                sim_df, x='cutoff', y='net_profit',
                title="Net Portfolio Yield Optimization Curve ($)",
                labels={'cutoff': 'Score Cutoff', 'net_profit': 'Projected Net Profit ($)'},
                color_discrete_sequence=['#16A34A']
            )
            fig_prof.add_vline(x=cutoff_val, line_dash="dash", line_color="#FFC107", annotation_text=f"Selected Cutoff: {cutoff_val}")
            st.plotly_chart(fig_prof, width="stretch")
            
    else:
        st.info("Lending simulation report not found.")

# ==========================================
# 8. Expected Credit Loss Framework
# ==========================================
elif nav_selection == "Expected Credit Loss Framework":
    st.header("Expected Credit Loss (ECL) Framework")
    st.markdown("""
    <div style="background-color: var(--secondary-background-color); border-left: 5px solid var(--primary-color); padding: 15px; border-radius: 6px; margin-bottom: 20px; font-size: 0.95rem; line-height: 1.5; color: var(--text-color);">
        <b>IFRS 9 / Basel Credit Analytics Integration:</b> Evaluate credit portfolio loss expectations by decomposing risk into Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). 
        Expected Credit Loss is calculated using the regulatory standard formula: 
        <span style="font-family: monospace; font-weight: bold; color: var(--primary-color);">ECL = PD × LGD × EAD</span>.
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Risk Decomposition Cards
    st.subheader("Credit Loss Component Decomposition")
    
    col_card1, col_card2, col_card3 = st.columns(3)
    
    with col_card1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(37, 99, 235, 0.01) 100%); border: 1px solid rgba(37, 99, 235, 0.2); border-radius: 12px; padding: 20px; min-height: 230px;">
            <div style="margin-bottom: 12px; display: flex; align-items: center;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M12 16v-4"></path>
                    <path d="M12 8h.01"></path>
                </svg>
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 700; color: var(--text-color);">Probability of Default (PD)</span>
            </div>
            <p style="font-size: 0.88rem; line-height: 1.5; color: var(--text-color); opacity: 0.85;">
                The mathematical likelihood that a borrower will default on their debt obligation over a 12-month horizon. 
            </p>
            <div style="margin-top: 15px; font-size: 0.8rem; color: var(--text-color); opacity: 0.7;">
                <b>Driven by:</b> Application Scorecard (Logistic Regression) based on age of history, income, inquiries, and utilization.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_card2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(249, 115, 22, 0.08) 0%, rgba(249, 115, 22, 0.01) 100%); border: 1px solid rgba(249, 115, 22, 0.2); border-radius: 12px; padding: 20px; min-height: 230px;">
            <div style="margin-bottom: 12px; display: flex; align-items: center;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F97316" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    <path d="M12 8v4"></path>
                    <path d="M12 16h.01"></path>
                </svg>
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 700; color: var(--text-color);">Loss Given Default (LGD)</span>
            </div>
            <p style="font-size: 0.88rem; line-height: 1.5; color: var(--text-color); opacity: 0.85;">
                The severity of loss if a default occurs, expressing net credit loss as a percentage of exposure. Calculated as <code>(Exposure - Recoveries) / Exposure</code>.
            </p>
            <div style="margin-top: 15px; font-size: 0.8rem; color: var(--text-color); opacity: 0.7;">
                <b>Driven by:</b> XGBoost regression on default cohorts, modeling recovery potential based on application-time assets.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_card3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.01) 100%); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 12px; padding: 20px; min-height: 230px;">
            <div style="margin-bottom: 12px; display: flex; align-items: center;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px;">
                    <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
                    <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
                </svg>
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 700; color: var(--text-color);">Exposure at Default (EAD)</span>
            </div>
            <p style="font-size: 0.88rem; line-height: 1.5; color: var(--text-color); opacity: 0.85;">
                The projected gross outstanding balance of the facility at the time of default. Accounts for principal repayment schedule prior to default.
            </p>
            <div style="margin-top: 15px; font-size: 0.8rem; color: var(--text-color); opacity: 0.7;">
                <b>Driven by:</b> XGBoost regression modeling the percent of the initial funded loan amount still unpaid when default takes place.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Check for ECL Engine
    ecl_engine = None
    try:
        from ecl_engine import ECLEngine
        ecl_engine = ECLEngine()
    except Exception as e:
        st.warning(f"Expected Credit Loss engine components are being calibrated. Please complete training first. (Error: {e})")
        
    if ecl_engine is not None:
        col_ecl1, col_ecl2 = st.columns([3, 2])
        
        with col_ecl1:
            st.subheader("Expected Loss Calculator")
            
            with st.container():
                st.write("### Borrower Financial & Credit Inputs")
                
                e_loan_amnt = st.number_input(
                    "Requested Loan Amount ($):",
                    min_value=500, max_value=100000, value=15000, step=500,
                    key="ecl_loan_amnt",
                    help="Determines baseline exposure. EAD is computed as EAD% x Loan Amount."
                )
                
                e_annual_inc = st.number_input(
                    "Annual Income ($):",
                    min_value=0, max_value=5000000, value=65000, step=1000,
                    key="ecl_annual_inc",
                    help="Higher income reduces probability of default (PD) score points."
                )
                
                e_dti = st.number_input(
                    "Debt-to-Income (DTI) Ratio (%):",
                    min_value=0.0, max_value=100.0, value=15.0, step=0.1,
                    key="ecl_dti"
                )
                
                e_revol_util = st.number_input(
                    "Revolving Line Utilization (%):",
                    min_value=0.0, max_value=150.0, value=45.0, step=1.0,
                    key="ecl_revol_util",
                    help="Revolving credit line utilization percentage. Higher values increase PD score."
                )
                
                e_inq = st.selectbox(
                    "Inquiries in Last 6 Months:",
                    [0, 1, 2, 3],
                    key="ecl_inq",
                    help="Number of credit inquiries in past 6 months. Drives PD score."
                )
                
                e_pub_rec = st.number_input(
                    "Public Derogatory Records:",
                    min_value=0, max_value=100, value=0, step=1,
                    key="ecl_pub_rec",
                    help="Number of derogatory public records on the credit history."
                )
                
                e_bankrupt = st.selectbox(
                    "Public Record Bankruptcies:",
                    [0.0, 1.0, 2.0],
                    key="ecl_bankrupt",
                    help="Bankruptcy status. Drives PD scorecard points."
                )
                
                e_purpose = st.selectbox(
                    "Loan Purpose:",
                    ["DEBT_CONSOLIDATION", "CREDIT_CARD", "HOME_IMPROVEMENT", "WEDDING", "CAR", "MAJOR_PURCHASE", "MEDICAL", "OTHER", "VACATION", "EDUCATIONAL", "HOUSE", "RENEWABLE_ENERGY", "SMALL_BUSINESS"],
                    key="ecl_purpose",
                    help="Stated purpose of the loan. Drives PD scorecard risk points."
                )
                
                e_home = st.selectbox(
                    "Home Ownership Status:",
                    ["MORTGAGE", "RENT", "OWN", "OTHER"],
                    key="ecl_home"
                )
                
                e_emp = st.number_input(
                    "Employment Length (Years):",
                    min_value=0, max_value=10, value=5, step=1,
                    key="ecl_emp"
                )
                
                e_delinq = st.number_input(
                    "Delinquencies in Past 2 Years:",
                    min_value=0, max_value=50, value=0, step=1,
                    key="ecl_delinq"
                )
                
                e_open_acc = st.number_input(
                    "Open Credit Accounts:",
                    min_value=0, max_value=100, value=10, step=1,
                    key="ecl_open_acc"
                )
                
                e_cr_age = st.number_input(
                    "Credit History Age (Months):",
                    min_value=0, max_value=1200, value=180, step=12,
                    key="ecl_cr_age"
                )
                
            # Pack input dictionary
            applicant_data = {
                "loan_amnt": e_loan_amnt,
                "funded_amnt": e_loan_amnt,
                "annual_inc": e_annual_inc,
                "dti": e_dti,
                "revol_util": e_revol_util,
                "inq_last_6mths": e_inq,
                "pub_rec": e_pub_rec,
                "pub_rec_bankruptcies": e_bankrupt,
                "purpose": e_purpose.lower(),
                "home_ownership": e_home,
                "emp_length": e_emp,
                "delinq_2yrs": e_delinq,
                "open_acc": e_open_acc,
                "credit_history_age": e_cr_age
            }
            
        with col_ecl2:
            st.subheader("Expected Credit Loss Diagnostics")
            
            # Predict
            res = ecl_engine.predict_applicant_ecl(applicant_data)
            
            # Extract
            score = res["credit_score"]
            pd_val = res["probability_of_default"]
            lgd_val = res["lgd_percentage"]
            ead_pct = res["ead_percentage"]
            ead_amt = res["ead_amount"]
            ecl_val = res["expected_credit_loss"]
            ecl_pct = res["expected_credit_loss_percentage"]
            tier = res["ecl_risk_tier"]
            pd_band = res["pd_risk_band"]
            
            # Colors based on Risk Tier
            if tier == "Low ECL":
                badge_bg = "#D1E7DD"
                badge_color = "#0F5132"
                border_color = "#A3CFBB"
            elif tier == "Medium ECL":
                badge_bg = "#FFF3CD"
                badge_color = "#664D03"
                border_color = "#FFE69C"
            else:
                badge_bg = "#F8D7DA"
                badge_color = "#842029"
                border_color = "#F5C2C7"
                
            st.markdown(clean_html(f"""
            <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); border-radius: 12px; padding: 25px; margin-bottom: 20px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-color); opacity: 0.6; font-weight: 600;">ECL Risk Assessment</div>
                <div style="display: inline-block; padding: 4px 12px; font-weight: 700; border-radius: 6px; font-size: 1.1rem; margin-top: 8px; margin-bottom: 20px; background-color: {badge_bg}; color: {badge_color}; border: 1px solid {border_color};">
                    {tier}
                </div>
                
                <div style="font-family: 'Outfit', sans-serif; font-size: 0.85rem; color: var(--text-color); opacity: 0.6; font-weight: 600; margin-bottom: 4px;">Expected Credit Loss (ECL)</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; color: {badge_color if tier == 'High ECL' else 'var(--text-color)'}; line-height: 1.1; margin-bottom: 25px;">
                    ${ecl_val:.2f}
                    <span style="font-size: 1.05rem; font-weight: 500; opacity: 0.7; color: var(--text-color);">({ecl_pct:.2%})</span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr; gap: 12px; border-top: 1px solid rgba(128,128,128,0.15); padding-top: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.88rem; color: var(--text-color); opacity: 0.75;">Credit Score & PD</span>
                        <span style="font-family: monospace; font-size: 0.95rem; font-weight: 700; color: var(--text-color);">{score} pts ({pd_val:.2%})</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.88rem; color: var(--text-color); opacity: 0.75;">Loss Given Default (LGD)</span>
                        <span style="font-family: monospace; font-size: 0.95rem; font-weight: 700; color: var(--text-color);">{lgd_val:.2%}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.88rem; color: var(--text-color); opacity: 0.75;">Exposure at Default (EAD)</span>
                        <span style="font-family: monospace; font-size: 0.95rem; font-weight: 700; color: var(--text-color);">{ead_pct:.2%} (${ead_amt:,.2f})</span>
                    </div>
                </div>
            </div>
            """), unsafe_allow_html=True)
            
            st.markdown("### Decision Recommendation")
            st.write(f"The baseline applicant scoring recommendation is **{res['pd_decision']}** based on their PD risk band (**{pd_band}**). Under the ECL Risk Tier system, this credit risk profile triggers a **{tier}** designation.")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Expected Credit Loss Flow (Sequencing credit risk)")
    
    st.markdown(clean_html("""
    <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); border-radius: 12px; padding: 25px; display: flex; justify-content: center; overflow-x: auto; margin-bottom: 25px;">
        <svg width="680" height="120" viewBox="0 0 680 120" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%;">
            <!-- Step 1: PD -->
            <rect x="10" y="10" width="130" height="70" rx="8" fill="#2563EB" stroke="#1D4ED8" stroke-width="2"/>
            <text x="75" y="38" font-family="'Inter', sans-serif" font-size="11" font-weight="700" fill="#FFFFFF" text-anchor="middle">Probability (PD)</text>
            <text x="75" y="58" font-family="'Inter', sans-serif" font-size="9" opacity="0.85" fill="#FFFFFF" text-anchor="middle">Underwriting Scorecard</text>
            
            <path d="M150 45 L 180 45" stroke="#64748B" stroke-width="2" marker-end="url(#arrow)"/>
            
            <!-- Step 2: EAD -->
            <rect x="190" y="10" width="130" height="70" rx="8" fill="#10B981" stroke="#059669" stroke-width="2"/>
            <text x="255" y="38" font-family="'Inter', sans-serif" font-size="11" font-weight="700" fill="#FFFFFF" text-anchor="middle">Exposure (EAD)</text>
            <text x="255" y="58" font-family="'Inter', sans-serif" font-size="9" opacity="0.85" fill="#FFFFFF" text-anchor="middle">Outstanding Balance</text>
            
            <path d="M330 45 L 360 45" stroke="#64748B" stroke-width="2" marker-end="url(#arrow)"/>
            
            <!-- Step 3: LGD -->
            <rect x="370" y="10" width="130" height="70" rx="8" fill="#F97316" stroke="#EA580C" stroke-width="2"/>
            <text x="435" y="38" font-family="'Inter', sans-serif" font-size="11" font-weight="700" fill="#FFFFFF" text-anchor="middle">Severity (LGD)</text>
            <text x="435" y="58" font-family="'Inter', sans-serif" font-size="9" opacity="0.85" fill="#FFFFFF" text-anchor="middle">Unrecovered Loss %</text>
            
            <path d="M510 45 L 540 45" stroke="#64748B" stroke-width="2" marker-end="url(#arrow)"/>
            
            <!-- Step 4: ECL -->
            <rect x="550" y="10" width="120" height="70" rx="8" fill="#7B61FF" stroke="#6366F1" stroke-width="2"/>
            <text x="610" y="38" font-family="'Inter', sans-serif" font-size="12" font-weight="800" fill="#FFFFFF" text-anchor="middle">ECL Value</text>
            <text x="610" y="58" font-family="'Inter', sans-serif" font-size="9" font-weight="700" fill="#FFFFFF" text-anchor="middle">PD × LGD × EAD</text>
            
            <!-- Definitions -->
            <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748B"/>
                </marker>
            </defs>
        </svg>
    </div>
    """), unsafe_allow_html=True)



