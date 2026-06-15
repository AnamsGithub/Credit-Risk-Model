"""
Generate a professional, LinkedIn-ready PDF presentation for the
Credit Risk Scorecard & PD Modeling System.

Focus tabs:
  1. Lending Policy Simulator
  2. Expected Credit Loss Framework
  3. Model Development Lifecycle
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import Flowable
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor, white, black, Color
import reportlab.lib.colors as rlcolors

# ─────────────────────────────────────────────
# BRAND PALETTE
# ─────────────────────────────────────────────
NAVY      = HexColor("#002B49")
BLUE      = HexColor("#2563EB")
CYAN      = HexColor("#0891B2")
TEAL      = HexColor("#0F766E")
GREEN     = HexColor("#059669")
GREEN_LT  = HexColor("#34D399")
AMBER     = HexColor("#D97706")
RED       = HexColor("#DC3545")
ROSE      = HexColor("#BE123C")
PURPLE    = HexColor("#7C3AED")
SLATE     = HexColor("#1E293B")
SLATE_MD  = HexColor("#334155")
SLATE_LT  = HexColor("#64748B")
BG_DARK   = HexColor("#0F172A")
BG_CARD   = HexColor("#1E293B")
BG_CARD2  = HexColor("#162032")
WHITE     = white
GREY_LT   = HexColor("#E2E8F0")
GREY_MD   = HexColor("#94A3B8")

W, H = A4  # 595.27 x 841.89

# ─────────────────────────────────────────────
# CUSTOM FLOWABLES
# ─────────────────────────────────────────────

class FullPageBackground(Flowable):
    """Draw a dark gradient-style background covering the whole page."""
    def __init__(self, width, height, top_color, bottom_color):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.top_color = top_color
        self.bottom_color = bottom_color

    def draw(self):
        # Simulate gradient with several bands
        steps = 20
        for i in range(steps):
            t = i / steps
            r = self.top_color.red   + t * (self.bottom_color.red   - self.top_color.red)
            g = self.top_color.green + t * (self.bottom_color.green - self.top_color.green)
            b = self.top_color.blue  + t * (self.bottom_color.blue  - self.top_color.blue)
            band_h = self.height / steps
            self.canv.setFillColorRGB(r, g, b)
            self.canv.rect(0, self.height - (i + 1) * band_h, self.width, band_h + 1, fill=1, stroke=0)


class ColoredRect(Flowable):
    """A simple filled rectangle (used for section headers, dividers)."""
    def __init__(self, width, height, color, radius=4):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
        self.radius = radius

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height, self.radius, fill=1, stroke=0)


class AccentLine(Flowable):
    """A thin colored horizontal rule."""
    def __init__(self, width, color, thickness=2):
        Flowable.__init__(self)
        self.width = width
        self.height = thickness
        self.color = color
        self.thickness = thickness

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.thickness, fill=1, stroke=0)


class MetricCard(Flowable):
    """Draw a metric card with label, value, and sub-label."""
    def __init__(self, width, height, label, value, sublabel, accent_color, bg_color=None):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.label = label
        self.value = value
        self.sublabel = sublabel
        self.accent = accent_color
        self.bg = bg_color or BG_CARD

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        # Left accent bar
        c.setFillColor(self.accent)
        c.roundRect(0, 0, 4, self.height, 3, fill=1, stroke=0)
        # Label
        c.setFillColor(GREY_MD)
        c.setFont("Helvetica", 7)
        c.drawString(12, self.height - 14, self.label.upper())
        # Value
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 17)
        c.drawString(12, self.height - 34, self.value)
        # Sub-label
        c.setFillColor(GREY_MD)
        c.setFont("Helvetica", 7)
        c.drawString(12, 8, self.sublabel)


class StepBox(Flowable):
    """A single step in the lifecycle stepper."""
    def __init__(self, width, step_num, title, description, accent_color):
        Flowable.__init__(self)
        self.width = width
        self.height = 52
        self.step_num = step_num
        self.title = title
        self.description = description
        self.accent = accent_color

    def draw(self):
        c = self.canv
        # Card background
        c.setFillColor(BG_CARD)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        # Circle
        c.setFillColor(self.accent)
        c.circle(22, self.height / 2, 12, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 9)
        num_str = str(self.step_num)
        c.drawCentredString(22, self.height / 2 - 4, num_str)
        # Title
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(42, self.height - 16, self.title)
        # Description
        c.setFillColor(GREY_MD)
        c.setFont("Helvetica", 7.5)
        # Wrap text manually
        words = self.description.split()
        line = ""
        y = self.height - 28
        for word in words:
            test = (line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 7.5) < self.width - 52:
                line = test
            else:
                c.drawString(42, y, line)
                y -= 10
                line = word
        if line:
            c.drawString(42, y, line)


class ECLFormulaBox(Flowable):
    """Draw the ECL = PD × LGD × EAD formula visually."""
    def __init__(self, width):
        Flowable.__init__(self)
        self.width = width
        self.height = 70

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(BG_CARD)
        c.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)
        # Draw formula boxes
        boxes = [
            ("ECL", "Expected Credit Loss", BLUE),
            ("=", "", SLATE_LT),
            ("PD", "Probability of Default", TEAL),
            ("×", "", SLATE_LT),
            ("LGD", "Loss Given Default", AMBER),
            ("×", "", SLATE_LT),
            ("EAD", "Exposure at Default", GREEN),
        ]
        total_w = self.width - 20
        box_w = total_w / 7
        x = 10
        for label, sub, color in boxes:
            if label in ("=", "×"):
                c.setFillColor(GREY_MD)
                c.setFont("Helvetica-Bold", 18)
                c.drawCentredString(x + box_w / 2, self.height / 2 - 7, label)
            else:
                c.setFillColor(color)
                c.roundRect(x, 10, box_w - 4, self.height - 20, 5, fill=1, stroke=0)
                c.setFillColor(WHITE)
                c.setFont("Helvetica-Bold", 12)
                c.drawCentredString(x + (box_w - 4) / 2, self.height / 2 + 2, label)
                c.setFont("Helvetica", 6)
                c.drawCentredString(x + (box_w - 4) / 2, 14, sub[:20])
            x += box_w


class RiskTierBar(Flowable):
    """Three-segment bar showing Low / Medium / High ECL risk tiers."""
    def __init__(self, width):
        Flowable.__init__(self)
        self.width = width
        self.height = 40

    def draw(self):
        c = self.canv
        segs = [
            (GREEN,  "LOW ECL",    "≤ 1.5% of Loan", 0.33),
            (AMBER,  "MEDIUM ECL", "1.5% – 5.0%",    0.34),
            (RED,    "HIGH ECL",   "> 5.0% of Loan",  0.33),
        ]
        x = 0
        for color, label, sub, pct in segs:
            w = self.width * pct
            c.setFillColor(color)
            c.roundRect(x, 18, w - 2, 16, 3, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 7)
            c.drawCentredString(x + (w - 2) / 2, 24, label)
            c.setFillColor(GREY_MD)
            c.setFont("Helvetica", 6.5)
            c.drawCentredString(x + (w - 2) / 2, 8, sub)
            x += w


class TradeoffBar(Flowable):
    """Simple visual showing approval vs default rate trade-off."""
    def __init__(self, width):
        Flowable.__init__(self)
        self.width = width
        self.height = 80

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(BG_CARD)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)

        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(12, self.height - 14, "Score Cutoff → Portfolio Trade-off")

        # At cutoff 620
        data = [
            ("600", 96.2, 15.9),
            ("620", 90.8, 14.6),
            ("640", 82.3, 13.1),
            ("660", 71.5, 11.8),
            ("680", 58.2, 10.2),
        ]
        bar_area_w = self.width - 24
        bar_max_h = 38
        x_start = 12
        bar_w = bar_area_w / len(data) - 4
        for i, (cutoff, approval, default_r) in enumerate(data):
            x = x_start + i * (bar_area_w / len(data))
            # Approval bar (blue)
            bh = bar_max_h * approval / 100
            c.setFillColor(BLUE)
            c.roundRect(x, 20, bar_w * 0.45, bh, 2, fill=1, stroke=0)
            # Default bar (red)
            bh2 = bar_max_h * default_r / 15
            c.setFillColor(RED)
            c.roundRect(x + bar_w * 0.5, 20, bar_w * 0.45, bh2, 2, fill=1, stroke=0)
            # Label
            c.setFillColor(GREY_MD)
            c.setFont("Helvetica", 6)
            c.drawCentredString(x + bar_w / 2, 12, cutoff)
            # Star at optimal 620
            if cutoff == "620":
                c.setFillColor(AMBER)
                c.setFont("Helvetica-Bold", 8)
                c.drawString(x, 60, "★ Optimal")

        # Legend
        c.setFillColor(BLUE)
        c.rect(self.width - 90, self.height - 16, 8, 6, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica", 6.5)
        c.drawString(self.width - 80, self.height - 15, "Approval Rate")
        c.setFillColor(RED)
        c.rect(self.width - 90, self.height - 26, 8, 6, fill=1, stroke=0)
        c.drawString(self.width - 80, self.height - 25, "Default Rate")
        c.setFillColor(WHITE)


# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=WHITE)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    styles = {
        "cover_title": S("cover_title", fontName="Helvetica-Bold", fontSize=32,
                         leading=38, alignment=TA_CENTER, textColor=WHITE),
        "cover_sub":   S("cover_sub",   fontName="Helvetica", fontSize=15,
                         leading=20, alignment=TA_CENTER, textColor=GREY_LT),
        "cover_tag":   S("cover_tag",   fontName="Helvetica", fontSize=10,
                         alignment=TA_CENTER, textColor=GREY_MD),
        "section_hdr": S("section_hdr", fontName="Helvetica-Bold", fontSize=18,
                         leading=22, textColor=WHITE),
        "section_sub": S("section_sub", fontName="Helvetica", fontSize=11,
                         leading=15, textColor=GREY_LT),
        "card_title":  S("card_title",  fontName="Helvetica-Bold", fontSize=11,
                         leading=14, textColor=WHITE),
        "body":        S("body",        fontName="Helvetica", fontSize=9,
                         leading=13, textColor=GREY_LT),
        "body_white":  S("body_white",  fontName="Helvetica", fontSize=9,
                         leading=13, textColor=WHITE),
        "bullet":      S("bullet",      fontName="Helvetica", fontSize=8.5,
                         leading=13, textColor=GREY_LT, leftIndent=10, bulletIndent=0),
        "small":       S("small",       fontName="Helvetica", fontSize=7.5,
                         leading=11, textColor=GREY_MD),
        "kpi_label":   S("kpi_label",   fontName="Helvetica-Bold", fontSize=7,
                         leading=10, textColor=GREY_MD),
        "tag":         S("tag",         fontName="Helvetica-Bold", fontSize=7.5,
                         leading=10, alignment=TA_CENTER, textColor=WHITE),
        "footer":      S("footer",      fontName="Helvetica", fontSize=7,
                         alignment=TA_CENTER, textColor=GREY_MD),
    }
    return styles

# ─────────────────────────────────────────────
# PAGE BUILDER HELPERS
# ─────────────────────────────────────────────

def page_background(canvas, doc):
    """Draw dark background on every page except cover."""
    canvas.saveState()
    # Dark gradient background
    steps = 30
    top_c = HexColor("#0A1628")
    bot_c = HexColor("#060D1A")
    for i in range(steps):
        t = i / steps
        r = top_c.red   + t * (bot_c.red   - top_c.red)
        g = top_c.green + t * (bot_c.green - top_c.green)
        b = top_c.blue  + t * (bot_c.blue  - top_c.blue)
        bh = H / steps
        canvas.setFillColorRGB(r, g, b)
        canvas.rect(0, H - (i + 1) * bh, W, bh + 1, fill=1, stroke=0)

    # Top accent bar
    canvas.setFillColor(BLUE)
    canvas.rect(0, H - 3, W, 3, fill=1, stroke=0)

    # Footer
    canvas.setFillColor(SLATE)
    canvas.rect(0, 0, W, 22, fill=1, stroke=0)
    canvas.setFillColor(GREY_MD)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W / 2, 7, "Credit Risk Scorecard & PD Modeling System  |  github.com/AnamsGithub/Credit-Risk-Model  |  Designed by Anam")
    canvas.restoreState()


def cover_background(canvas, doc):
    """Full dark gradient with accent geometry for cover page."""
    canvas.saveState()
    # Deep dark navy
    steps = 40
    top_c = HexColor("#061022")
    bot_c = HexColor("#020810")
    for i in range(steps):
        t = i / steps
        r = top_c.red   + t * (bot_c.red   - top_c.red)
        g = top_c.green + t * (bot_c.green - top_c.green)
        b = top_c.blue  + t * (bot_c.blue  - top_c.blue)
        bh = H / steps
        canvas.setFillColorRGB(r, g, b)
        canvas.rect(0, H - (i + 1) * bh, W, bh + 1, fill=1, stroke=0)

    # Big decorative circles
    canvas.setFillColor(HexColor("#0A1E3D"))
    canvas.circle(W * 1.1, H * 0.75, 200, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#071428"))
    canvas.circle(W * -0.1, H * 0.3, 160, fill=1, stroke=0)

    # Top accent stripe
    canvas.setFillColor(BLUE)
    canvas.rect(0, H - 4, W, 4, fill=1, stroke=0)

    # Blue diagonal accent bar
    canvas.setFillColor(HexColor("#0B2347"))
    p = canvas.beginPath()
    p.moveTo(0, H * 0.55)
    p.lineTo(W * 0.45, H * 0.45)
    p.lineTo(W * 0.45, H * 0.35)
    p.lineTo(0, H * 0.45)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)

    # Bottom bar
    canvas.setFillColor(SLATE)
    canvas.rect(0, 0, W, 30, fill=1, stroke=0)
    canvas.setFillColor(GREY_MD)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(W / 2, 10, "github.com/AnamsGithub/Credit-Risk-Model  |  Designed & Developed by Anam  |  Machine Learning | Risk Analytics | Business Strategy")
    canvas.restoreState()


def section_header_block(title, subtitle, accent_color, styles):
    """Returns story elements for a section header."""
    elems = []
    elems.append(Spacer(1, 8))
    elems.append(AccentLine(W - 72, accent_color, thickness=3))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(title, styles["section_hdr"]))
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(subtitle, styles["section_sub"]))
    elems.append(Spacer(1, 14))
    return elems


def tag_pill(text, color, styles):
    data = [[Paragraph(text, styles["tag"])]]
    t = Table(data, colWidths=[80], rowHeights=[16])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("ROUNDEDCORNERS", [4]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


# ─────────────────────────────────────────────
# BUILD STORY
# ─────────────────────────────────────────────

def build_story(styles):
    story = []
    inner_w = W - 72  # page margins 36 each side

    # ══════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════
    story.append(Spacer(1, 90))
    story.append(Paragraph("Credit Risk Scorecard", styles["cover_title"]))
    story.append(Paragraph("& PD Modeling System", styles["cover_title"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph("An Institutional-Grade Credit Decisioning & Capital Provisioning Platform", styles["cover_sub"]))
    story.append(Spacer(1, 28))

    # Tech tags
    tags_data = [[
        tag_pill("Python", TEAL, styles),
        tag_pill("XGBoost", BLUE, styles),
        tag_pill("Basel II/III", NAVY, styles),
        tag_pill("IFRS 9", PURPLE, styles),
        tag_pill("Streamlit", GREEN, styles),
        tag_pill("Plotly", AMBER, styles),
    ]]
    tags_tbl = Table(tags_data, colWidths=[80] * 6, hAlign="CENTER")
    tags_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tags_tbl)
    story.append(Spacer(1, 36))

    # Cover KPI row
    metrics_data = [[
        MetricCard(110, 65, "ROC-AUC", "0.6311", "OOT Validation",    BLUE),
        MetricCard(110, 65, "Gini",    "26.22%", "Risk Separation",    PURPLE),
        MetricCard(110, 65, "PSI",     "0.0091", "✓ Stable",           GREEN),
        MetricCard(110, 65, "Net Yield", "$65M", "Optimal Cutoff 620", AMBER),
    ]]
    metrics_tbl = Table(metrics_data, colWidths=[110] * 4, hAlign="CENTER")
    metrics_tbl.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(metrics_tbl)
    story.append(Spacer(1, 30))

    # Cover feature highlights
    highlights = [
        ("📊", "WoE Scorecard",    "Logistic Regression + integer points (300–850)"),
        ("🤖", "XGBoost Challenger", "98.5% predictive parity with full explainability"),
        ("⚖️", "ECL Framework",    "PD × LGD × EAD — Basel/IFRS 9 compliant"),
        ("📈", "Policy Optimizer", "Real-time trade-off curves & net margin projections"),
        ("🔍", "Drift Monitoring", "PSI & CSI population stability tracking"),
        ("📱", "Live Dashboard",   "8-tab enterprise portal, mobile-responsive"),
    ]
    h_data = [[Paragraph(f"{e} <b>{t}</b><br/><font size=7 color='#94A3B8'>{d}</font>", styles["body_white"])
               for e, t, d in highlights[:3]],
              [Paragraph(f"{e} <b>{t}</b><br/><font size=7 color='#94A3B8'>{d}</font>", styles["body_white"])
               for e, t, d in highlights[3:]]]
    h_tbl = Table(h_data, colWidths=[160, 160, 160], hAlign="CENTER")
    h_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BG_CARD, BG_CARD2]),
        ("ROUNDEDCORNERS", [6]),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1E3A5F")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(h_tbl)
    story.append(Spacer(1, 30))
    story.append(Paragraph("Designed & Developed by Anam  ·  Machine Learning | Risk Analytics | Business Strategy", styles["cover_tag"]))

    # ══════════════════════════════════════════
    # PAGE 2 — MODEL DEVELOPMENT LIFECYCLE
    # ══════════════════════════════════════════
    story.append(Spacer(1, 20))
    story += section_header_block(
        "🔬  Model Development Lifecycle",
        "A rigorous, 9-stage analytical pipeline from raw data ingestion to automated credit policy execution.",
        TEAL, styles
    )

    steps = [
        (1, "Data Ingestion & Target Definition", "Ingested 39,717 loan records. Defined Default (1) = Charged Off/Defaulted, Non-Default (0) = Fully Paid. Removed active loans to eliminate classification bias.", BLUE),
        (2, "Out-of-Time Cohort Segmentation", "Enforced temporal split — In-Time training (2007–2010, 18,061 loans) vs. Out-of-Time validation (2011, 20,516 loans) to simulate real-world future deployment.", CYAN),
        (3, "Coarse Binning & WoE Mapping", "Binned raw variables via Decision Tree cuts. Computed Weight of Evidence (WoE) to convert categories into risk log-odds ensuring monotonic default relationships.", TEAL),
        (4, "Feature Selection via Information Value", "Calculated IV for all variables. Retained 6 core predictors (0.02 ≤ IV ≤ 0.50): revol_util, purpose, inq_last_6mths, annual_inc, pub_rec, pub_rec_bankruptcies.", GREEN),
        (5, "Logistic Scorecard Scaling", "Fit WoE predictors using Logistic Regression. Scaled odds to FICO-style integers (Base Score 600 at 50:1 odds, PDO = 20, Range 300–850) suitable for banking systems.", GREEN_LT),
        (6, "XGBoost Challenger Benchmark", "Constructed gradient-boosted tree model (XGBoost) on raw variables as a challenger — providing a non-linear performance baseline without regulatory interpretability.", AMBER),
        (7, "Model Validation & Verification", "Benchmarked on OOT cohort: ROC-AUC 0.6311 (Scorecard) vs 0.6331 (XGBoost). Brier Score 0.1308 vs 0.2274 — Scorecard significantly better calibrated for capital adequacy.", RED),
        (8, "Stability & Drift Monitoring", "Quantified population shifts using PSI (0.0091 — Stable ✓) and CSI. Flagged pub_rec_bankruptcies drift (CSI = 0.286) indicating 2011 bankruptcy profile changes.", ROSE),
        (9, "Lending Policy Cutoff Simulation", "Simulated ECL, approval volumes, interest yields, and net profits across score thresholds to identify the optimal business lending policy (Cutoff 620).", PURPLE),
    ]

    for i, (num, title, desc, color) in enumerate(steps):
        story.append(StepBox(inner_w, num, title, desc, color))
        if i < len(steps) - 1:
            story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "★  Champion Scorecard retains 98.5% of XGBoost's predictive power while delivering full regulatory transparency — making it the production choice.",
        ParagraphStyle("note", fontName="Helvetica-Oblique", fontSize=8, leading=11,
                       textColor=AMBER, leftIndent=4, rightIndent=4)
    ))

    # ══════════════════════════════════════════
    # PAGE 3 — EXPECTED CREDIT LOSS FRAMEWORK
    # ══════════════════════════════════════════
    story.append(Spacer(1, 20))
    story += section_header_block(
        "⚖️  Expected Credit Loss (ECL) Framework",
        "IFRS 9 & Basel-aligned capital provisioning — combining PD, LGD, and EAD into a unified credit loss engine.",
        PURPLE, styles
    )

    # ECL Formula
    story.append(ECLFormulaBox(inner_w))
    story.append(Spacer(1, 14))

    # Three pillars
    pillars = [
        ("PD", "Probability of Default",
         "The likelihood that a borrower will default within the forecast horizon. "
         "Derived from the Champion Logistic Scorecard (Brier 0.1308) — superior calibration "
         "ensures PD estimates match real default rates, critical for IFRS 9 staging.",
         TEAL),
        ("LGD", "Loss Given Default",
         "The fraction of exposure lost if a borrower defaults. "
         "Modeled via XGBoost Regression on the defaulted cohort (LGD = [Exposure − Recoveries] / Exposure). "
         "Capped [0,1]. Benchmarked against RandomForest. Avg LGD: ~60%.",
         AMBER),
        ("EAD", "Exposure at Default",
         "The expected outstanding principal at time of default. "
         "Modeled as EAD% = [Funded Amount − Principal Repaid] / Funded Amount using XGBoost Regressor. "
         "Capped [0,1]. Accounts for amortization schedule prior to default.",
         GREEN),
    ]
    pillar_col_w = (inner_w - 10) / 3
    pillar_data = [[]]
    for short, full, desc, color in pillars:
        cell_content = [
            Paragraph(f'<font size=18 color="{color.hexval() if hasattr(color, "hexval") else "#059669"}"><b>{short}</b></font>', styles["section_hdr"]),
            Spacer(1, 3),
            Paragraph(f"<b>{full}</b>", styles["card_title"]),
            Spacer(1, 4),
            Paragraph(desc, styles["small"]),
        ]
        pillar_data[0].append(cell_content)

    def make_pillar_table(pillars_list, col_w):
        cells = []
        for short, full, desc, color in pillars_list:
            hex_color = "#{:02X}{:02X}{:02X}".format(
                int(color.red * 255), int(color.green * 255), int(color.blue * 255)
            )
            txt = (
                f'<font size=16 color="{hex_color}"><b>{short}</b></font><br/>'
                f'<font size=9><b>{full}</b></font><br/><br/>'
                f'<font size=7.5 color="#94A3B8">{desc}</font>'
            )
            cells.append(Paragraph(txt, styles["body_white"]))
        tbl = Table([cells], colWidths=[col_w] * 3)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), BG_CARD),
            ("COLBACKGROUNDS", (0, 0), (-1, -1), [BG_CARD, BG_CARD2, BG_CARD]),
            ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#1E3A5F")),
            ("TOPPADDING",    (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("ROUNDEDCORNERS", [6]),
        ]))
        return tbl

    story.append(make_pillar_table(pillars, pillar_col_w))
    story.append(Spacer(1, 14))

    # ECL Risk Tiers
    story.append(Paragraph("<b>ECL Risk Tier Classification</b>", styles["card_title"]))
    story.append(Spacer(1, 6))
    story.append(RiskTierBar(inner_w))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "ECL is computed per application at run-time. Risk tier drives lending decisions, capital reserves, and IFRS 9 Stage assignment (Stage 1 / Stage 2 / Stage 3).",
        styles["small"]
    ))
    story.append(Spacer(1, 14))

    # Regulatory alignment box
    reg_txt = (
        "<b>Regulatory Alignment:</b>  "
        "IFRS 9 requires banks to provision Expected Credit Losses across 12-month (Stage 1) and lifetime horizons (Stage 2/3). "
        "Basel II/III mandates PD × LGD × EAD as the core IRB capital requirement formula. "
        "This platform implements both standards end-to-end — from model training to live ECL calculation — "
        "supporting capital adequacy reporting, loan provisioning, and risk-based pricing."
    )
    reg_data = [[Paragraph(reg_txt, styles["small"])]]
    reg_tbl = Table(reg_data, colWidths=[inner_w])
    reg_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#1A2744")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEAFTER",     (0, 0), (0, -1), 4, PURPLE),
    ]))
    story.append(reg_tbl)

    # ══════════════════════════════════════════
    # PAGE 4 — LENDING POLICY SIMULATOR
    # ══════════════════════════════════════════
    story.append(Spacer(1, 20))
    story += section_header_block(
        "📈  Lending Policy Simulator",
        "An interactive cutoff optimizer — dynamically balance approval rates, default risk, and portfolio net margin in real time.",
        BLUE, styles
    )

    # What it does
    story.append(Paragraph(
        "The Lending Policy Simulator is the <b>strategic command centre</b> of the platform. "
        "Risk officers slide a score cutoff threshold and instantly see the downstream impact on every key portfolio metric. "
        "No more static Excel models — this is live, model-powered scenario analysis.",
        styles["body_white"]
    ))
    story.append(Spacer(1, 12))

    # Trade-off chart
    story.append(TradeoffBar(inner_w))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Approval Rate (blue) vs Default Rate (red) across score cutoffs. ★ marks the optimal business cutoff.", styles["small"]))
    story.append(Spacer(1, 14))

    # Optimal cutoff metrics grid
    story.append(Paragraph("<b>Optimal Cutoff — Score 620</b>", styles["card_title"]))
    story.append(Spacer(1, 8))

    card_w = (inner_w - 15) / 4
    cards_row1 = [
        MetricCard(card_w, 62, "Approval Rate",    "90.8%",    "Target Underwriting", BLUE),
        MetricCard(card_w, 62, "Default Rate",     "14.60%",   "Expected Bad Rate",   RED),
        MetricCard(card_w, 62, "Approved Loans",   "18,628",   "OOT Accounts",        PURPLE),
        MetricCard(card_w, 62, "Approved Volume",  "$222.96M", "Loan Exposure",       CYAN),
    ]
    cards_row2 = [
        MetricCard(card_w, 62, "Interest Yield",   "$97.97M",  "Gross Revenue",       AMBER),
        MetricCard(card_w, 62, "Credit Loss (ECL)", "$32.93M", "Expected Loss",       ROSE),
        MetricCard(card_w, 62, "Net Profit Yield", "$65.04M",  "Portfolio Net Margin", GREEN),
        MetricCard(card_w, 62, "vs Baseline",      "+$8.2M",   "Policy Alpha Generated", TEAL),
    ]

    cards_tbl1 = Table([cards_row1], colWidths=[card_w] * 4, rowHeights=[62])
    cards_tbl1.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cards_tbl1)
    story.append(Spacer(1, 6))
    cards_tbl2 = Table([cards_row2], colWidths=[card_w] * 4, rowHeights=[62])
    cards_tbl2.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cards_tbl2)
    story.append(Spacer(1, 14))

    # Key insight box
    insight_items = [
        "📉  Raising cutoff from 580 → 620 reduces default rate by 1.27pp while sacrificing only 5.4% approval volume.",
        "💰  Each 10-point score increase generates approximately $4M additional net margin at optimal exposure levels.",
        "⚠️   Aggressive cutoffs (>660) cannibalize $100M+ in loan volume with diminishing returns on default reduction.",
        "🎯  Cutoff 620 is the Pareto-optimal policy: maximum net margin at controlled default rates.",
    ]
    insight_data = [[Paragraph(item, styles["bullet"])] for item in insight_items]
    insight_tbl = Table(insight_data, colWidths=[inner_w])
    insight_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BG_CARD),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BG_CARD, BG_CARD2]),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.3, HexColor("#1E3A5F")),
        ("LINEAFTER",     (0, 0), (0, -1), 3, BLUE),
    ]))
    story.append(insight_tbl)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Parameters: Avg Interest Rate 12.0% · LGD 60% · Loan Term 3 Years · Data: 20,516 OOT Applications",
        styles["small"]
    ))

    # ══════════════════════════════════════════
    # PAGE 5 — CLOSING / CTA
    # ══════════════════════════════════════════
    story.append(Spacer(1, 30))
    story += section_header_block(
        "🚀  Explore the Platform",
        "An end-to-end credit risk analytics suite — built for banking rigor, designed for modern data teams.",
        GREEN, styles
    )

    # 8-tab summary
    tabs = [
        ("1", "Executive Overview",           "6 KPIs, portfolio distribution, model scorecard",  BLUE),
        ("2", "Credit Risk Simulator",        "Live underwriting: decision + scorecard breakdown",  CYAN),
        ("3", "Lending Policy Simulator",     "Cutoff optimizer, profit curves, ECL projections",  TEAL),
        ("4", "Expected Credit Loss Framework","PD×LGD×EAD engine, IFRS 9 risk tiers",             GREEN),
        ("5", "Model Development Lifecycle",  "9-stage visual stepper from data to policy",        GREEN_LT),
        ("6", "Understanding Credit Risk",    "Layman's guide: WoE, IV, scorecards explained",     AMBER),
        ("7", "Champion vs Challenger",       "ROC, KS, calibration curves side-by-side",          RED),
        ("8", "Portfolio Stability",          "PSI matrix, CSI drift tracking, score migrations",  PURPLE),
    ]
    tab_col_w = (inner_w - 8) / 2
    tab_rows = []
    for i in range(0, len(tabs), 2):
        row = []
        for j in range(2):
            if i + j < len(tabs):
                num, name, desc, color = tabs[i + j]
                hex_c = "#{:02X}{:02X}{:02X}".format(int(color.red*255), int(color.green*255), int(color.blue*255))
                txt = f'<font color="{hex_c}"><b>Tab {num} — {name}</b></font><br/><font size=7.5 color="#94A3B8">{desc}</font>'
                row.append(Paragraph(txt, styles["body_white"]))
            else:
                row.append(Paragraph("", styles["body_white"]))
        tab_rows.append(row)

    tabs_tbl = Table(tab_rows, colWidths=[tab_col_w, tab_col_w])
    tabs_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BG_CARD),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BG_CARD, BG_CARD2]),
        ("GRID",          (0, 0), (-1, -1), 0.3, HexColor("#1E3A5F")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tabs_tbl)
    story.append(Spacer(1, 20))

    # CTA Links
    cta_txt = (
        "<b>🔗 GitHub Repository:</b>  github.com/AnamsGithub/Credit-Risk-Model<br/><br/>"
        "<b>📊 Live Dashboard:</b>  [YOUR_SITE_URL]<br/><br/>"
        "<b>👤 Designed & Developed by:</b>  Anam  |  Machine Learning · Risk Analytics · Business Strategy"
    )
    cta_data = [[Paragraph(cta_txt, styles["body_white"])]]
    cta_tbl = Table(cta_data, colWidths=[inner_w])
    cta_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#0D1F3C")),
        ("LINEABOVE",     (0, 0), (-1, 0), 3, BLUE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
        ("TOPPADDING",    (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(cta_tbl)

    return story


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def generate_pdf(output_path="outputs/Credit_Risk_Platform_Presentation.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use first page template for cover, second template for content
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=28,
    )

    styles = make_styles()
    story = build_story(styles)

    def first_page(canvas, doc):
        cover_background(canvas, doc)

    def later_pages(canvas, doc):
        page_background(canvas, doc)

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    print(f"✅  PDF generated: {output_path}")


if __name__ == "__main__":
    generate_pdf()
