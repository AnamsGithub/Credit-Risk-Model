# Credit Risk Scorecard Dashboard Deployment Guide

This guide provides instructions to deploy the interactive credit risk scorecard dashboard to public hosting platforms.

---

## Deployment Option 1: Hugging Face Spaces (Recommended - Free & Instant)

Hugging Face Spaces is the easiest way to deploy python-based Streamlit apps.

### Step 1: Create a Hugging Face Account & Space
1. Go to [Hugging Face](https://huggingface.co/) and create a free account.
2. Click on **Spaces** in the top navigation, then click **Create new Space**.
3. Configure your Space:
   * **Space Name:** `credit-risk-scorecard` (or any custom name)
   * **SDK:** Select **Streamlit**
   * **Space Hardware:** Select **Cpu basic (Free)**
   * **Visibility:** Select **Public**
4. Click **Create Space**.

### Step 2: Upload Files to the Space
You can clone the HF repository via Git LFS and push files, or upload them directly using the HF Web UI:
1. Upload the following files and folders preserving the folder hierarchy:
   * `dashboard/app.py` -> Upload as `app.py` in the root of the space.
   * `score_generation_engine.py`
   * `configs/config.yaml`
   * `requirements.txt`
   * `outputs/scorecards/scorecard_table.csv`
   * `outputs/scorecards/woe_tables.csv`
   * `outputs/scorecards/iv_report.csv`
   * `outputs/monitoring/champion_challenger_report.xlsx`
   * `outputs/monitoring/psi_report.xlsx`
   * `outputs/monitoring/csi_report.xlsx`
   * `outputs/monitoring/score_monitoring_report.xlsx`
   * `outputs/reports/` (all generated `.png` and `.html` curve files)
2. Note: You can exclude the notebook directories and the raw `loan.csv` file since the production engine does not need the raw data (it relies purely on pre-calculated scorecard lookup tables!).

### Step 3: Wait for Build
Once files are uploaded, Hugging Face will read `requirements.txt`, install dependencies, compile the container, and launch your Streamlit app automatically in less than 2 minutes!

---

## Deployment Option 2: Render (Free Web Service)

Render allows you to deploy applications directly from a GitHub repository.

### Step 1: Push Your Project to GitHub
1. Create a public repository on GitHub named `credit-risk-scorecard`.
2. Commit and push your local files (you can exclude `data/loan.csv` to keep the repo clean).
3. Ensure `requirements.txt` is present in the root directory.

### Step 2: Create a Web Service on Render
1. Sign up/Log in to [Render](https://render.com/).
2. Click **New** -> **Web Service**.
3. Connect your GitHub account and select your `credit-risk-scorecard` repository.
4. Configure the service:
   * **Name:** `credit-risk-scorecard`
   * **Environment:** Select **Python**
   * **Region:** Select a region close to you
   * **Branch:** `main`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
   * **Instance Type:** Select **Free**
5. Click **Create Web Service**. Render will build the repository and launch the app. The URL will be displayed at the top of the Render console.
