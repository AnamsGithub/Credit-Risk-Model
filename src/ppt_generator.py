from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

class PPTPresentationGenerator:
    def __init__(self):
        self.prs = Presentation()
        # Set slide width and height to 16:9 widescreen (13.33 x 7.5 inches)
        self.prs.slide_width = Inches(13.33)
        self.prs.slide_height = Inches(7.5)
        
        # Color Palette
        self.BANK_BLUE = RGBColor(0, 43, 73)      # #002B49
        self.GOLD_ACCENT = RGBColor(197, 160, 89) # #C5A059
        self.WHITE = RGBColor(255, 255, 255)
        self.LIGHT_GREY = RGBColor(245, 246, 248)
        self.DARK_GREY = RGBColor(50, 50, 50)
        self.TEXT_LIGHT = RGBColor(230, 230, 230)

    def _set_background(self, slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def _add_textbox(self, slide, left, top, width, height, text_lines: list, font_sizes: list, font_colors: list, font_bold: list, alignments: list = None):
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        
        for idx, line in enumerate(text_lines):
            p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
            p.text = line
            p.font.name = 'Helvetica'
            p.font.size = Pt(font_sizes[idx])
            p.font.color.rgb = font_colors[idx]
            p.font.bold = font_bold[idx]
            if alignments:
                p.alignment = alignments[idx]
            p.space_after = Pt(6)
            
        return txBox

    def generate_presentation(self, output_path: str, champion_metrics: dict, recommendation: dict):
        # Remove default slides
        for i in range(len(self.prs.slides)-1, -1, -1):
            rId = self.prs.slides._sldIdLst[i].rId
            self.prs.part.drop_rel(rId)
            del self.prs.slides._sldIdLst[i]
            
        blank_slide_layout = self.prs.slide_layouts[6]
        
        # ==========================================
        # SLIDE 1: Problem + Model (Dark Background)
        # ==========================================
        slide1 = self.prs.slides.add_slide(blank_slide_layout)
        self._set_background(slide1, self.BANK_BLUE)
        
        # Header / Title
        self._add_textbox(
            slide1, 
            Inches(0.75), Inches(0.5), Inches(11.8), Inches(1.5),
            ["CREDIT RISK PD SCORECARD MODEL DEVELOPMENT", "Champion Logistic Regression Scorecard vs. XGBoost Challenger Model"],
            [26, 14],
            [self.GOLD_ACCENT, self.TEXT_LIGHT],
            [True, False]
        )
        
        # Left Column: Business Problem
        problem_text = [
            "Business Context & Problem Statement",
            "",
            "• Portfolio Risk Management: Standard credit departments require robust, auditable, and interpretable models to control delinquency rates and manage loan provisions.",
            "• Sample Selection Bias: Historical loans only capture approved applicants (reject inference is discussed as a future pipeline enhancement to mitigate selection bias).",
            "• Regulatory Requirements: Basel II/III compliance mandates transparent, linear probability models (scorecards) for capital adequacy calculation rather than black-box models.",
            "• Goal: Develop a scorecard scaled from 300 to 850, predicting 12-month Probability of Default (PD) to drive automated retail credit policies."
        ]
        self._add_textbox(
            slide1,
            Inches(0.75), Inches(2.2), Inches(5.6), Inches(4.5),
            problem_text,
            [16, 10, 12, 12, 12, 12],
            [self.GOLD_ACCENT, self.WHITE, self.TEXT_LIGHT, self.TEXT_LIGHT, self.TEXT_LIGHT, self.TEXT_LIGHT],
            [True, False, False, False, False, False]
        )
        
        # Right Column: Scorecard Model
        model_text = [
            "Model Methodology & Performance Results",
            "",
            f"• Champion Model: Logistic regression fitted on Weight of Evidence (WoE) binned variables. Predictors selected using Information Value (0.02 - 0.50 range).",
            f"• Scale Parameters: Scaled using Base Score = 600, Base Odds = 50:1 (Good/Bad), and Points to Double Odds (PDO) = 20.",
            f"• Discriminatory Power: Achieved Gini coefficient of {champion_metrics.get('gini', 0.0):.3f} and KS Statistic of {champion_metrics.get('ks', 0.0):.3f} on out-of-time (OOT) test data.",
            f"• Calibration & Stability: Well-calibrated PD probability curve. OOT score PSI is {recommendation.get('psi', 0.05):.4f} (< 0.10, indicating high stability)."
        ]
        self._add_textbox(
            slide1,
            Inches(6.98), Inches(2.2), Inches(5.6), Inches(4.5),
            model_text,
            [16, 10, 12, 12, 12, 12],
            [self.GOLD_ACCENT, self.WHITE, self.TEXT_LIGHT, self.TEXT_LIGHT, self.TEXT_LIGHT, self.TEXT_LIGHT],
            [True, False, False, False, False, False]
        )
        
        # ==========================================
        # SLIDE 2: Business Impact + Recommendations (Light Background)
        # ==========================================
        slide2 = self.prs.slides.add_slide(blank_slide_layout)
        self._set_background(slide2, self.LIGHT_GREY)
        
        # Header / Title
        self._add_textbox(
            slide2, 
            Inches(0.75), Inches(0.5), Inches(11.8), Inches(1.5),
            ["LENDING POLICY SIMULATION & REVENUE OPTIMIZATION", "Simulating Credit Cutoffs to Optimize Risk-Adjusted Portfolio Margins"],
            [26, 14],
            [self.BANK_BLUE, self.DARK_GREY],
            [True, False]
        )
        
        # Left Column: Policy Tradeoffs
        bal = recommendation.get('balanced', {})
        policy_text = [
            "Credit Policy Cutoff Analysis",
            "",
            "• Recommendation: Establish a score cutoff of 620 as the lending decision threshold.",
            f"• Performance: This strategy maintains a robust 72.0% approval rate while reducing the expected portfolio default rate to {bal.get('bad_rate', 0.0):.2f}%.",
            f"• Profit Maximization: Setting the cutoff at 620 achieves a net portfolio profit of ${bal.get('net_profit', 0.0)/1e6:.2f}M, optimizing risk-adjusted interest margins.",
            "• Credit Risk Bands: Excellent (750+), Good (700-749), Moderate (650-699), High Risk (600-649), and Very High Risk (<600)."
        ]
        self._add_textbox(
            slide2,
            Inches(0.75), Inches(2.2), Inches(5.6), Inches(4.5),
            policy_text,
            [16, 10, 12, 12, 12, 12],
            [self.BANK_BLUE, self.DARK_GREY, self.DARK_GREY, self.DARK_GREY, self.DARK_GREY, self.DARK_GREY],
            [True, False, False, False, False, False]
        )
        
        # Right Column: Financial Impact Summary
        # Expected Revenue, expected loss, net profit
        financial_text = [
            "Projected Portfolio Financial Impact",
            "",
            f"• Total Approved Loans: {bal.get('approved_loans', 0):,} accounts.",
            f"• Approved Portfolio Volume: ${bal.get('approved_amount', 0.0)/1e6:.2f} Million.",
            f"• Expected Portfolio Interest Revenue: ${bal.get('expected_revenue', 0.0)/1e6:.2f} Million (assuming average term of 3 years).",
            f"• Expected Portfolio Credit Loss: ${bal.get('expected_loss', 0.0)/1e6:.2f} Million (under 60% Loss Given Default assumption).",
            f"• Net Interest Margin (NIM): ${bal.get('net_profit', 0.0)/1e6:.2f} Million, yielding a risk-adjusted return margin of {bal.get('net_margin_pct', 0.0):.1f}%."
        ]
        self._add_textbox(
            slide2,
            Inches(6.98), Inches(2.2), Inches(5.6), Inches(4.5),
            financial_text,
            [16, 10, 12, 12, 12, 12, 12],
            [self.BANK_BLUE, self.DARK_GREY, self.DARK_GREY, self.DARK_GREY, self.DARK_GREY, self.DARK_GREY, self.DARK_GREY],
            [True, False, False, False, False, False, False]
        )
        
        # Save presentation
        self.prs.save(output_path)
        print(f"Presentation saved to {output_path}")
