import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go

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
# Load Model & Scaler
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_ml_assets()
except Exception as e:
    st.error(f"Error loading ML models: {e}")
    st.stop()

FEATURE_NAMES = [
    "account_length",
    "area_code",
    "international_plan",
    "voice_mail_plan",
    "total_day_calls",
    "total_day_charge",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_calls",
    "total_night_charge",
    "total_intl_calls",
    "total_intl_charge",
    "number_customer_service_calls"
]

# ---------------------------------------------------------
# Modern Executive Glassmorphism Theme (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main App Background */
    .stApp {
        background: linear-gradient(180deg, #0b132b 0%, #1c2541 100%);
        color: #f8fafc;
    }

    /* Glassmorphic Top Header */
    .app-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 28px 36px;
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
        margin-bottom: 10px;
    }

    .header-title {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-desc {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Glass Card Container */
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
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .field-desc {
        font-size: 12px;
        color: #94a3b8;
        margin-top: -6px;
        margin-bottom: 12px;
        line-height: 1.3;
    }

    /* Importance Badges */
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

    /* Risk Outcome Glass Cards */
    .risk-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 28px;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }

    .risk-card-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.25) 100%);
        border: 2px solid #f59e0b;
        border-radius: 20px;
        padding: 28px;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);
    }

    .risk-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.25) 100%);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 28px;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }

    .prob-val {
        font-size: 54px;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1;
        margin: 12px 0;
    }

    /* Diagnosis Pill */
    .diag-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
        font-size: 13px;
        color: #e2e8f0;
    }

    /* Modern Styled Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        color: #ffffff;
        border-radius: 14px;
        height: 52px;
        font-size: 16px;
        font-weight: 700;
        border: none;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Glass Header
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1 class="header-title">Telecom Customer Churn Predictor</h1>
    <p class="header-desc">Simple tool for non-technical users. Adjust the customer details below to see if a customer is likely to leave or stay.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Application Main Grid Layout
# ---------------------------------------------------------
col_inputs, col_analytics = st.columns([1.1, 1.3], gap="large")

with col_inputs:
    st.markdown("""
    <div class="card-title">
        <span>🔥 Key Customer Risk Signals</span>
        <span class="impact-high">#1 Priority</span>
    </div>
    """, unsafe_allow_html=True)

    # 1. Customer Service Calls
    service_calls = st.slider(
        "🎧 Customer Support Calls (Last 30 Days)",
        min_value=0, max_value=10, value=4,
        help="How many times the customer contacted customer care/support in the last month."
    )
    st.markdown("<div class='field-desc'>💡 <strong>Why this matters:</strong> Customers who call support 4+ times are often frustrated with network or billing issues and are at highest risk of leaving.</div>", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        intl_plan_sel = st.selectbox(
            "✈️ International Calling Plan",
            options=["No International Plan", "Active International Plan"],
            index=1,
            help="Does the customer have an international calling plan addon?"
        )
        international_plan = 1 if "Active" in intl_plan_sel else 0
        st.markdown("<div class='field-desc'>💡 <strong>Why this matters:</strong> International callers churn more often (42%) if international rates or connection quality are poor.</div>", unsafe_allow_html=True)

    with col_p2:
        vmail_plan_sel = st.selectbox(
            "📬 Voicemail Plan",
            options=["No Voicemail Plan", "Active Voicemail Plan"],
            index=0,
            help="Does the customer subscribe to voicemail?"
        )
        voice_mail_plan = 1 if "Active" in vmail_plan_sel else 0
        st.markdown("<div class='field-desc'>💡 <strong>Why this matters:</strong> Customers using voicemail service tend to be more attached to their number and leave less often.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 2. Usage & Spending Sliders
    st.markdown("""
    <div class="card-title">
        <span>💰 Billing & Account Duration</span>
        <span class="impact-med">Bill Factors</span>
    </div>
    """, unsafe_allow_html=True)

    day_charge = st.slider(
        "☀️ Daytime Call Charges ($/month)",
        min_value=0.0, max_value=70.0, value=45.0, step=1.0,
        help="Total monthly dollar amount billed for calls during daytime peak hours."
    )
    st.markdown("<div class='field-desc'>💡 <strong>Why this matters:</strong> Daytime charges make up the largest part of the bill. High bills cause price sensitivity.</div>", unsafe_allow_html=True)

    tenure_months = st.slider(
        "⏳ Customer Tenure (Months with us)",
        min_value=1, max_value=240, value=36, step=1,
        help="How long (in months) this customer has been subscribing to our service."
    )
    st.markdown("<div class='field-desc'>💡 <strong>Why this matters:</strong> Brand-new customers (< 12 months) leave more easily than long-time loyal subscribers.</div>", unsafe_allow_html=True)

    area_code = st.selectbox(
        "📍 Customer Region Area Code",
        options=[408, 415, 510],
        index=1,
        format_func=lambda x: f"Area Code {x}",
        help="The regional area code for the customer's phone number."
    )
    st.markdown("<div class='field-desc'>💡 <strong>Why this matters:</strong> Helps identify if network issues or competitor offers in specific regions are causing churn.</div>", unsafe_allow_html=True)

    # 3. Collapsible Advanced Usage Details (Auto-calculated intelligent defaults)
    with st.expander("⚙️ Advanced Usage Breakdown (Optional for Detailed Analysis)", expanded=False):
        st.caption("ℹ️ Non-technical users can skip this! These secondary call details are automatically calculated for you:")
        
        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            total_day_calls = st.number_input("Daytime Calls Count", min_value=0, max_value=300, value=110, help="Number of daytime calls made.")
            total_eve_calls = st.number_input("Evening Calls Count", min_value=0, max_value=300, value=100, help="Number of evening calls made.")
            total_eve_charge = st.number_input("Evening Call Charge ($)", min_value=0.0, max_value=100.0, value=20.0, step=0.5, help="Dollar cost of evening calls.")
        with adv_c2:
            total_night_calls = st.number_input("Night Calls Count", min_value=0, max_value=300, value=95, help="Number of night calls made.")
            total_night_charge = st.number_input("Night Call Charge ($)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, help="Dollar cost of night calls.")
            total_intl_calls = st.number_input("Int'l Calls Count", min_value=0, max_value=50, value=3, help="Number of international calls made.")
            total_intl_charge = st.number_input("Int'l Call Charge ($)", min_value=0.0, max_value=50.0, value=4.5, step=0.5, help="Dollar cost of international calls.")

    # Construct input dataframe
    input_dict = {
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
    
    input_df = pd.DataFrame([[input_dict[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)
    
    st.markdown(" ")
    predict_btn = st.button("🚀 Calculate Customer Churn Risk", use_container_width=True)

# ---------------------------------------------------------
# Right Column: Real-Time Diagnostic & Recommendation Engine
# ---------------------------------------------------------
with col_analytics:
    st.markdown("""
    <div class="card-title">
        <span>🎯 Prediction Result & AI Recommendation</span>
    </div>
    """, unsafe_allow_html=True)

    # Run Prediction
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob_churn = model.predict_proba(input_scaled)[0][1]

    # Risk Tiering
    if prob_churn >= 0.65:
        risk_tier = "HIGH CHURN RISK"
        card_css = "risk-card-high"
        gauge_color = "#ef4444"
        icon = "🚨"
        summary_text = "Action Needed: This customer is very likely to cancel their telecom service soon."
    elif prob_churn >= 0.35:
        risk_tier = "MEDIUM CHURN RISK"
        card_css = "risk-card-med"
        gauge_color = "#f59e0b"
        icon = "⚠️"
        summary_text = "Warning Sign: Customer shows moderate dissatisfaction signals. Contact them soon."
    else:
        risk_tier = "LOW CHURN RISK"
        card_css = "risk-card-low"
        gauge_color = "#10b981"
        icon = "💚"
        summary_text = "Safe Account: Customer is happy, stable, and unlikely to leave."

    # Top Row: Outcome Card & Plotly Gauge Meter
    res_col1, res_col2 = st.columns([1.1, 1])

    with res_col1:
        st.markdown(f"""
        <div class="{card_css}">
            <div style="font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                {icon} {risk_tier}
            </div>
            <div class="prob-val">{prob_churn:.1%}</div>
            <div style="font-size: 13px; opacity: 0.95; line-height: 1.4;">
                {summary_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        # Plotly Speedometer Meter
        fig_meter = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_churn * 100,
            number={'suffix': "%", 'font': {'size': 26, 'color': '#ffffff', 'weight': 800}},
            title={'text': "Risk Meter", 'font': {'size': 12, 'color': '#94a3b8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': gauge_color, 'thickness': 0.35},
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
        fig_meter.update_layout(
            height=190,
            margin=dict(l=15, r=15, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_meter, use_container_width=True)

    # Key Risk Driver Explanations ("Why this customer is at risk")
    st.markdown("### ❓ Why is this customer at risk?")
    
    reasons = []
    if service_calls >= 4:
        reasons.append(f"❌ <strong>High Support Escalations ({service_calls} calls)</strong>: Customer has called support repeatedly, indicating unresolved network or bill issues.")
    elif service_calls >= 2:
        reasons.append(f"⚠️ <strong>Elevated Support Activity ({service_calls} calls)</strong>: Customer reached out to support multiple times recently.")

    if international_plan == 1 and total_intl_charge > 3.0:
        reasons.append(f"❌ <strong>High International Spending (${total_intl_charge:.2f})</strong>: Customer pays extra for int'l calls, which can cause bill shock.")

    if day_charge > 40.0:
        reasons.append(f"❌ <strong>High Daytime Call Charges (${day_charge:.2f}/mo)</strong>: Above-average monthly daytime bill increases likelihood of switching.")

    if tenure_months < 12:
        reasons.append(f"⚠️ <strong>New Customer Account ({tenure_months} months)</strong>: Customer is relatively new and has not built long-term loyalty yet.")

    if not reasons:
        reasons.append("💚 <strong>Low Risk Customer Pattern</strong>: Customer service calls are low, bill is stable, and account history is solid.")

    for reason in reasons:
        st.markdown(f"""
        <div class="diag-box">
            {reason}
        </div>
        """, unsafe_allow_html=True)

    # Actionable AI Retention Recommendations
    st.markdown("### 💡 Easy Actions to Keep This Customer")
    
    actions = []
    if service_calls >= 3:
        actions.append("👨‍💼 <strong>Assign Dedicated Care Specialist</strong>: Have a senior support agent call the customer to resolve any open tickets.")
    if international_plan == 1 and total_intl_charge > 3.0:
        actions.append("✈️ <strong>Offer Int'l Discount Pass</strong>: Provide a 20% discount on international calling rates for the next 6 months.")
    if day_charge > 40.0:
        actions.append("💳 <strong>Offer Unlimited Call Plan Upgrade</strong>: Switch customer to an unlimited day plan to cap their monthly bill.")
    if tenure_months < 12 and prob_churn > 0.40:
        actions.append("🎁 <strong>New Subscriber Loyalty Bonus</strong>: Offer a $15 bill credit on their next invoice as a thank-you gesture.")
    if not actions:
        actions.append("💚 <strong>Maintain Regular Relationship</strong>: No urgent offer needed. Keep customer updated on standard loyalty perks.")

    for act in actions:
        st.markdown(f"- {act}", unsafe_allow_html=True)

    # Export Diagnostic Summary
    st.markdown("---")
    export_df = input_df.copy()
    export_df["predicted_churn"] = prediction
    export_df["churn_probability"] = prob_churn
    export_df["risk_tier"] = risk_tier
    export_df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_out = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Easy Summary Report (CSV)",
        data=csv_out,
        file_name=f"customer_churn_report_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------
# Minimal Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 12px;">
        📡 Telecom Customer Churn AI Predictor • Simple Executive Assistant for Customer Teams
    </div>
    """,
    unsafe_allow_html=True
)