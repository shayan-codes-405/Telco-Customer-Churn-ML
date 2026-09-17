# Telecom Customer Churn Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikitlearn)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render)

Live demo: https://telco-churn-70pb.onrender.com/

This project predicts telecom customer churn using a Random Forest model trained on usage, billing, plan, and support data. It identifies at-risk customers, explains the strongest churn drivers, and supports retention decisions. The app includes a FastAPI backend, a Streamlit dashboard, and a static web interface for live scoring and batch CSV analysis.

---

## Overview

This application analyzes telecom customer behavior to forecast churn risk using historical service usage, billing, plan activity, and support contact patterns. The system is designed for both business and technical users: a frontend dashboard helps explain risk factors, while the API provides a reusable inference layer for production use.

The deployed app reflects the real dashboard structure in this repository, including churn analytics, feature importance, service-call impact, and retention strategy insight cards.

---

## Features

- Real-time customer churn prediction
- Random Forest model-based scoring
- Dashboard for churn rate, retention metrics, and feature importance
- Support call and plan impact analysis
- CSV batch scoring for multiple customers
- FastAPI REST API with Swagger docs
- Streamlit interactive version for local exploration

---

## Project structure

```text
Telco-Customers-Churn/
├── app.py                     # Streamlit dashboard
├── server.py                  # FastAPI backend and static serving
├── index.html                 # Root landing page
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment configuration
├── README.md                  # Project documentation
├── data/
│   ├── raw/
│   │   ├── Data_Churn.csv
│   │   └── Data_Test.csv
│   └── processed/
│       ├── final.csv
│       └── submit.csv
├── frontend/
│   └── index.html
├── images/
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── notebooks/
│   └── Predict_Churn.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── predictor.py
├── static/
│   └── index.html
└── .gitignore
```

---

## Tech stack

- Python 3.10+
- FastAPI
- Streamlit
- scikit-learn
- Pandas and NumPy
- Render deployment

---

## Local setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Run the API server:

```bash
uvicorn server:app --reload --port 8000
```

Then open:

- Frontend: http://localhost:8501
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## Deployment

The project is deployed live on Render:

- https://telco-churn-70pb.onrender.com/

The backend serves the dashboard directly and exposes the ML API and analytics endpoints.

---

## API endpoints

- GET /health
- GET /api/info
- GET /api/analytics
- POST /api/predict
- POST /api/predict/batch
- GET /api/sample-csv

---

## Project notes

The model uses telecom customer attributes such as account length, area code, plan status, call volume, charge totals, and monthly service-call count to estimate churn probability. The repository includes preprocessing logic and model inference support in the src folder, which keeps the ML pipeline separated from the web app and makes it easier to maintain and extend.
