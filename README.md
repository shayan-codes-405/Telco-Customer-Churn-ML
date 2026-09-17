# 📡 Telecom Customer Churn AI Diagnostic Center

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikitlearn)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready Machine Learning web application and high-performance **FastAPI** backend that predicts telecom customer churn risk in real-time. Delivers instant risk scoring, explainable AI risk drivers, automated retention recommendations, and bulk CSV dataset scoring with an executive Glassmorphism dashboard.

---

## 📁 Clean Project Directory Structure

```
Telco-Customers-Churn/
│
├── data/
│   ├── raw/                      # Original raw datasets (Data_Churn.csv, Data_Test.csv)
│   └── processed/                # Evaluated outputs (final.csv, submit.csv)
│
├── models/                       # Trained ML artifacts
│   ├── model.pkl                 # Random Forest Classifier
│   ├── scaler.pkl                # Standard Scaler
│   └── label_encoder.pkl
│
├── notebooks/
│   └── Predict_Churn.ipynb       # Exploratory Data Analysis & Model Training
│
├── src/                          # Modular Python Inference & Engine Package
│   ├── __init__.py
│   ├── config.py                 # Feature definitions, paths & categorical mapping dictionaries
│   └── predictor.py              # Fixed ChurnPredictor class (resolves area_code distortion)
│
├── static/                       # Production Glassmorphism Web Dashboard
│   └── index.html                # Single customer diagnostics + Batch CSV upload dropzone
│
├── frontend/                     # Dedicated folder for Vercel deployment
│   └── index.html
│
├── images/                       # EDA charts & graphs
├── server.py                     # High-performance FastAPI REST API
├── app.py                        # Streamlit Dashboard (alternative UI)
├── requirements.txt              # Production dependencies
├── vercel.json                   # Vercel deployment configuration
├── render.yaml                   # Render web service configuration
└── README.md
```

---

## 🚀 Quick Start (Local Run)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Dashboard (Recommended)
```bash
uvicorn server:app --reload --port 8000
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.
- Interactive Dashboard: `http://localhost:8000`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`

### 3. Alternative: Run Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🌐 100% Free Cloud Deployment Guide (Render + Vercel)

### Option A: All-in-One Deployment on Render.com (Easiest)
Because FastAPI directly serves the static dashboard at `/`, you can host both the backend API and frontend on a single free Render Web Service:

1. Push your repository to GitHub.
2. Log in to **[Render.com](https://render.com/)** and click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Plan:** `Free`
5. Click **Deploy Web Service**. Your app will be live at `https://your-service.onrender.com`.

---

### Option B: Decoupled Architecture (Vercel Frontend + Render Backend)

#### Step 1: Deploy Backend on Render
1. Deploy the repository to Render as described in Option A.
2. Note your backend URL (e.g. `https://telco-churn-api.onrender.com`).

#### Step 2: Deploy Frontend to Vercel
1. Log in to **[Vercel.com](https://vercel.com/)** and click **Add New...** > **Project**.
2. Select your repository.
3. In Build & Development Settings:
   - **Framework Preset:** `Other`
   - **Root Directory:** `./` (or `static`)
4. Click **Deploy**.
5. Once deployed, open your Vercel site, go to the **⚙️ API & Deployment Config** tab, paste your Render URL (`https://telco-churn-api.onrender.com`), and click **Save & Test**.

---

### ⏰ How to Keep Render Backend Awake 24/7 (Prevent Sleep Mode)
Free Render web services sleep after 15 minutes of inactivity. To keep your API hot and fast:
1. Go to **[cron-job.org](https://cron-job.org/)** or **[UptimeRobot](https://uptimerobot.com/)** (both 100% free).
2. Create a new HTTP monitor pointing to:
   ```
   https://your-service.onrender.com/health
   ```
3. Set execution interval to **Every 10 minutes**.
4. Your API will now stay alive 24/7 with zero cold boot delays!

---

## 🛠️ API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server & model health status for uptime monitoring |
| `GET` | `/api/info` | Feature list and model metadata |
| `POST` | `/api/predict` | Single customer churn risk diagnostic |
| `POST` | `/api/predict/batch` | Bulk CSV file upload and scoring |
| `GET` | `/api/sample-csv` | Download sample CSV for testing |

---

## 🎯 Bug Fixes Applied
1. **Area Code Scaling Fix:** Training data mapped `area_code` (`area_code_408`, `area_code_415`, `area_code_510`) to `0, 1, 2`. The new inference pipeline maps area codes before standard scaling, preventing distorted inputs.
2. **Modular Architecture:** Extracted all inference and data preparation logic into `src/predictor.py` and `src/config.py`.
