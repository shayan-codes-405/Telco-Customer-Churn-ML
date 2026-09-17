import io
import time
import datetime
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from src.config import (
    FEATURE_NAMES,
    STATIC_DIR,
    DEFAULT_CUSTOMER,
    DATA_RAW_DIR
)
from src.predictor import ChurnPredictor

# Initialize FastAPI application
app = FastAPI(
    title="Telecom Churn AI Diagnostic Center API",
    description="High-performance machine learning inference API for telecom customer churn risk analysis.",
    version="2.0.0"
)

# Enable CORS for external frontends (e.g., Vercel, Netlify, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton on startup
predictor = ChurnPredictor()
start_time = time.time()


# Pydantic schema for single customer prediction
class CustomerInput(BaseModel):
    account_length: int = Field(default=36, description="Customer tenure in months with service", ge=1, le=300)
    area_code: Any = Field(default=415, description="Area code (408, 415, 510 or 'area_code_415')")
    international_plan: Any = Field(default=1, description="1 / 0 or 'yes' / 'no' for international plan")
    voice_mail_plan: Any = Field(default=0, description="1 / 0 or 'yes' / 'no' for voicemail plan")
    total_day_calls: int = Field(default=110, description="Total daytime calls count", ge=0, le=500)
    total_day_charge: float = Field(default=45.0, description="Total daytime charges in USD", ge=0.0)
    total_eve_calls: int = Field(default=100, description="Total evening calls count", ge=0, le=500)
    total_eve_charge: float = Field(default=20.0, description="Total evening charges in USD", ge=0.0)
    total_night_calls: int = Field(default=95, description="Total night calls count", ge=0, le=500)
    total_night_charge: float = Field(default=10.0, description="Total night charges in USD", ge=0.0)
    total_intl_calls: int = Field(default=3, description="Total international calls count", ge=0, le=100)
    total_intl_charge: float = Field(default=4.5, description="Total international charges in USD", ge=0.0)
    number_customer_service_calls: int = Field(default=4, description="Number of customer support inquiries", ge=0, le=20)

    class Config:
        json_schema_extra = {
            "example": DEFAULT_CUSTOMER
        }


# ---------------------------------------------------------------------------
# Health & Status Endpoints (Crucial for Keep-Alive pings on Render/Koyeb)
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """
    Health check endpoint for Render/UptimeRobot keep-alive pings.
    """
    uptime_seconds = int(time.time() - start_time)
    return {
        "status": "healthy",
        "service": "telecom-churn-api",
        "timestamp": datetime.datetime.now().isoformat(),
        "uptime_seconds": uptime_seconds,
        "model_loaded": predictor.model is not None,
        "scaler_loaded": predictor.scaler is not None
    }


@app.get("/api/info")
def get_api_info():
    """
    Returns feature specifications and default configuration.
    """
    return {
        "name": "Telecom Churn AI Diagnostic Center",
        "version": "2.0.0",
        "features": FEATURE_NAMES,
        "area_codes": [408, 415, 510],
        "default_customer": DEFAULT_CUSTOMER
    }


@app.get("/api/analytics")
def get_dataset_analytics():
    """
    Returns pre-computed and dynamic chart data from the training dataset
    and model feature importances for rich visual graphs.
    """
    try:
        data_path = DATA_RAW_DIR / "Data_Churn.csv"
        if not data_path.exists():
            raise FileNotFoundError("Data_Churn.csv not found.")
        
        df = pd.read_csv(data_path)
        
        # 1. Churn Ratio
        churn_counts = df['churn'].value_counts().to_dict()
        retained_count = int(churn_counts.get('no', 0))
        churn_count = int(churn_counts.get('yes', 0))
        total_count = len(df)
        churn_rate = round((churn_count / total_count) * 100, 1)

        # 2. Churn by Customer Service Calls
        calls_crosstab = pd.crosstab(df['number_customer_service_calls'], df['churn'], normalize='index') * 100
        service_call_labels = [f"{int(c)} calls" for c in calls_crosstab.index]
        service_call_churn_rates = [round(float(calls_crosstab.loc[c, 'yes']), 1) if 'yes' in calls_crosstab.columns else 0.0 for c in calls_crosstab.index]

        # 3. Churn by Plans
        intl_crosstab = pd.crosstab(df['international_plan'], df['churn'], normalize='index') * 100
        vmail_crosstab = pd.crosstab(df['voice_mail_plan'], df['churn'], normalize='index') * 100
        
        plan_churn = {
            "intl_with_plan": round(float(intl_crosstab.loc['yes', 'yes']), 1) if 'yes' in intl_crosstab.index else 42.2,
            "intl_without_plan": round(float(intl_crosstab.loc['no', 'yes']), 1) if 'no' in intl_crosstab.index else 11.2,
            "vmail_with_plan": round(float(vmail_crosstab.loc['yes', 'yes']), 1) if 'yes' in vmail_crosstab.index else 7.4,
            "vmail_without_plan": round(float(vmail_crosstab.loc['no', 'yes']), 1) if 'no' in vmail_crosstab.index else 16.4,
        }

        # 4. Model Feature Importances
        if hasattr(predictor.model, 'feature_importances_'):
            importances = predictor.model.feature_importances_
            feature_imp_list = [
                {"feature": feat, "importance": round(float(imp) * 100, 2)}
                for feat, imp in zip(FEATURE_NAMES, importances)
            ]
            feature_imp_list.sort(key=lambda x: x["importance"], reverse=True)
        else:
            feature_imp_list = []

        # 5. Day Charges Distribution by Churn
        retained_charges = df[df['churn'] == 'no']['total_day_charge'].tolist()
        churn_charges = df[df['churn'] == 'yes']['total_day_charge'].tolist()

        return {
            "success": True,
            "kpis": {
                "total_customers": total_count,
                "retained_count": retained_count,
                "churn_count": churn_count,
                "churn_rate_pct": f"{churn_rate}%",
                "model_type": "Random Forest Classifier (Balanced)",
                "active_features_count": len(FEATURE_NAMES)
            },
            "churn_ratio": {
                "labels": ["Retained (Active)", "Churned (Lost)"],
                "values": [retained_count, churn_count],
                "colors": ["#10b981", "#ef4444"]
            },
            "service_calls_churn": {
                "labels": service_call_labels,
                "rates": service_call_churn_rates
            },
            "plans_impact": plan_churn,
            "feature_importances": feature_imp_list,
            "charge_stats": {
                "retained_avg_day_charge": round(float(np.mean(retained_charges)), 2),
                "churn_avg_day_charge": round(float(np.mean(churn_charges)), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Prediction Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def predict_single_customer(customer: CustomerInput):
    """
    Predicts churn probability, risk level, risk drivers, and retention actions for a single customer.
    """
    try:
        data_dict = customer.dict()
        result = predictor.predict_single(data_dict)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict/batch")
async def predict_batch_customers(file: UploadFile = File(...)):
    """
    Upload a CSV file of customers for bulk prediction.
    Returns executive KPI metrics and scored records.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
            
        scored_df, summary = predictor.predict_batch(df)
        preview_records = scored_df.head(100).to_dict(orient="records")
        
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        return {
            "success": True,
            "filename": file.filename,
            "summary": summary,
            "preview": preview_records,
            "total_rows": len(scored_df),
            "csv_data": csv_string
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@app.get("/api/sample-csv")
def download_sample_csv():
    """
    Provides a sample CSV dataset for rapid testing in the UI.
    """
    test_csv_path = DATA_RAW_DIR / "Data_Test.csv"
    if test_csv_path.exists():
        return FileResponse(
            test_csv_path,
            media_type="text/csv",
            filename="sample_telecom_test_customers.csv"
        )
    else:
        df_sample = pd.DataFrame([DEFAULT_CUSTOMER])
        csv_buffer = io.StringIO()
        df_sample.to_csv(csv_buffer, index=False)
        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sample_customer.csv"}
        )


# ---------------------------------------------------------------------------
# Static Web Dashboard Mount (Serves the UI seamlessly)
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_dashboard():
    """
    Serves the primary Glassmorphism interactive dashboard.
    """
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Telecom Customer Churn AI API is online. Visit /docs for interactive Swagger API documentation."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
