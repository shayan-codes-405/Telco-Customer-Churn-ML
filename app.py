import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from src.config import FEATURE_NAMES, AREA_CODE_MAP, BINARY_MAP
from src.predictor import ChurnPredictor

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn AI Diagnostic Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Load Inference Engine (Cached)
# ---------------------------------------------------------
@st.cache_resource
def get_predictor():
    return ChurnPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading ML models: {e}")
    st.stop()

# ---------------------------------------------------------
# Modern Executive Glassmorphism Theme (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0b132b 0%, #1c2541 100%);
        color: #f8fafc;
    }

    .app-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .header-badge {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: #ffffff;
        padding: 4px 14px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }

    .header-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-desc {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .field-desc {
        font-size: 12px;
        color: #94a3b8;
        margin-top: -4px;
        margin-bottom: 12px;
        line-height: 1.3;
    }

    .impact-high {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }

    .impact-med {
        background: rgba(245, 158, 11, 0.2);
        color: #fde047;
        border: 1px solid rgba(245, 158, 11, 0.4);
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }

    .risk-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 24px;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }

    .risk-card-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.25) 100%);
        border: 2px solid #f59e0b;
        border-radius: 20px;
        padding: 24px;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);
    }

    .risk-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.25) 100%);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 24px;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }

    .prob-val {
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1;
        margin: 10px 0;
    }

    .diag-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 13px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Glass Header
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <span class="header-badge">AI Diagnostic Engine 2.0</span>
    <h1 class="header-title">Telecom Customer Churn Diagnostic Center</h1>
    <p class="header-desc">Real-time risk scoring, explainable AI risk drivers, and automated retention recommendations.</p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["⚡ Live Customer Diagnostic", "📁 Batch CSV Scoring"])

with tab_single:
    col_inputs, col_analytics = st.columns([1.1, 1.3], gap="large")

    with col_inputs:
        st.markdown("""
        <div class="card-title">
            <span>🔥 Key Customer Risk Signals</span>
            <span class="impact-high">#1 Priority</span>
        </div>
        """, unsafe_allow_html=True)

        service_calls = st.slider(
            "🎧 Customer Support Calls (Last 30 Days)",
            min_value=0, max_value=10, value=4,
            help="Number of times customer reached support in the last month."
        )
        st.markdown("<div class='field-desc'>💡 <strong>Insight:</strong> 4+ support calls indicates unresolved technical or billing dissatisfaction.</div>", unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            intl_plan_sel = st.selectbox(
                "✈️ International Calling Plan",
                options=["Active International Plan", "No International Plan"],
                index=0
            )
            international_plan = 1 if "Active" in intl_plan_sel else 0

        with col_p2:
            vmail_plan_sel = st.selectbox(
                "📬 Voicemail Plan",
                options=["No Voicemail Plan", "Active Voicemail Plan"],
                index=0
            )
            voice_mail_plan = 1 if "Active" in vmail_plan_sel else 0

        st.markdown("---")
        st.markdown("""
        <div class="card-title">
            <span>💰 Billing & Account Duration</span>
            <span class="impact-med">Bill Factors</span>
        </div>
        """, unsafe_allow_html=True)

        day_charge = st.slider(
            "☀️ Daytime Call Charges ($/month)",
            min_value=0.0, max_value=70.0, value=45.0, step=0.5
        )

        tenure_months = st.slider(
            "⏳ Customer Tenure (Months with us)",
            min_value=1, max_value=240, value=36, step=1
        )

        area_code = st.selectbox(
            "📍 Customer Region Area Code",
            options=[415, 408, 510],
            index=0,
            format_func=lambda x: f"Area Code {x}"
        )

        with st.expander("⚙️ Advanced Usage Breakdown (Optional)", expanded=False):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                total_day_calls = st.number_input("Daytime Calls Count", min_value=0, max_value=300, value=110)
                total_eve_calls = st.number_input("Evening Calls Count", min_value=0, max_value=300, value=100)
                total_eve_charge = st.number_input("Evening Call Charge ($)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
            with adv_c2:
                total_night_calls = st.number_input("Night Calls Count", min_value=0, max_value=300, value=95)
                total_night_charge = st.number_input("Night Call Charge ($)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
                total_intl_calls = st.number_input("Int'l Calls Count", min_value=0, max_value=50, value=3)
                total_intl_charge = st.number_input("Int'l Call Charge ($)", min_value=0.0, max_value=50.0, value=4.5, step=0.5)

        input_payload = {
            "account_length": tenure_months,
            "area_code": area_code,
            "international_plan": international_plan,
            "voice_mail_plan": voice_mail_plan,
            "total_day_calls": total_day_calls,
            "total_day_charge": day_charge,
            "total_eve_calls": total_eve_calls,
            "total_eve_charge": total_eve_charge,
            "total_night_calls": total_night_calls,
            "total_night_charge": total_night_charge,
            "total_intl_calls": total_intl_calls,
            "total_intl_charge": total_intl_charge,
            "number_customer_service_calls": service_calls
        }

    with col_analytics:
        st.markdown("""
        <div class="card-title">
            <span>🎯 Real-Time AI Diagnosis</span>
        </div>
        """, unsafe_allow_html=True)

        diag = predictor.predict_single(input_payload)
        prob_churn = diag["churn_probability"]
        risk_tier = diag["risk_tier"]
        card_css = "risk-card-high" if prob_churn >= 0.65 else ("risk-card-med" if prob_churn >= 0.35 else "risk-card-low")
        icon = "🚨" if prob_churn >= 0.65 else ("⚠️" if prob_churn >= 0.35 else "💚")

        res_col1, res_col2 = st.columns([1.1, 1])

        with res_col1:
            st.markdown(f"""
            <div class="{card_css}">
                <div style="font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">
                    {icon} {risk_tier}
                </div>
                <div class="prob-val">{diag['churn_probability_pct']}</div>
                <div style="font-size: 13px; opacity: 0.95;">
                    {diag['summary_text']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            fig_meter = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_churn * 100,
                number={'suffix': "%", 'font': {'size': 26, 'color': '#ffffff', 'weight': 800}},
                title={'text': "Risk Meter", 'font': {'size': 12, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                    'bar': {'color': diag['risk_color'], 'thickness': 0.35},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ]
                }
            ))
            fig_meter.update_layout(height=180, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_meter, use_container_width=True)

        st.markdown("### ❓ Key Risk Drivers")
        for driver in diag["risk_drivers"]:
            st.markdown(f"""
            <div class="diag-box">
                {driver['icon']} <strong>{driver['title']}</strong><br/>
                <span style="font-size: 12px; opacity: 0.85;">{driver['description']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 💡 AI Retention Actions")
        for act in diag["retention_actions"]:
            st.markdown(f"- **[{act['priority']}] {act['title']}**: {act['action']}")

        # Download Report
        report_df = pd.DataFrame([{**input_payload, **{"churn_probability": prob_churn, "risk_tier": risk_tier}}])
        csv_bytes = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Diagnostic Report (CSV)",
            data=csv_bytes,
            file_name=f"customer_churn_diagnosis_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

with tab_batch:
    st.markdown("### 📁 Upload Customer CSV Dataset")
    uploaded_file = st.file_uploader("Upload CSV dataset containing customer records", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            scored_df, summary = predictor.predict_batch(df_upload)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Customers", summary["total_records"])
            kpi2.metric("High Risk Customers", f"{summary['high_risk_count']} ({summary['high_risk_pct']})")
            kpi3.metric("Medium Risk", summary["medium_risk_count"])
            kpi4.metric("Avg Churn Probability", summary["avg_churn_probability"])

            st.dataframe(scored_df.head(100), use_container_width=True)

            scored_csv = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Complete Scored CSV",
                data=scored_csv,
                file_name=f"scored_customers_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")