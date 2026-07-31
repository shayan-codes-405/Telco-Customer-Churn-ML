import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import datetime

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
      background-color: #0d6efd;
      color: white;
      border-radius: 10px;
      height: 48px;
      font-size: 16px;
      font-weight: 700;
    }
    div.stButton > button:first-child:hover {
      background-color: #0b5ed7;
      color: white;
    }
    .big-probability {
      font-size: 36px;
      font-weight: 800;
      margin: 8px 0 0;
    }
    .risk-pill {
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      color: white;
      font-size: 16px;
      font-weight: 700;
      margin-top: 10px;
    }
    .result-card {
      padding: 18px;
      border-radius: 18px;
      box-shadow: 0 10px 20px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("# 📡 Telecom Customer Churn Prediction System")
st.markdown("Enter customer details in a simple form and get a churn prediction in seconds.")
st.markdown("---")


def friendly_yes_no(value):
    return "Yes" if value == 1 else "No"


def friendly_area_label(value):
    labels = {408: "Area 408", 415: "Area 415", 510: "Area 510"}
    return labels.get(value, str(value))


with st.sidebar:
    st.header("👤 Customer Details")
    st.caption("Fill in the form below using simple, everyday wording.")
    st.info("Tip: If you are unsure about a value, keep the suggested default.")

    account_length = st.number_input(
        "How long has the customer been with us? (months)",
        min_value=0,
        value=100,
        step=1,
        help="Enter the number of months the customer has been with the company."
    )
    area_code = st.selectbox(
        "Which region is the customer in?",
        options=[408, 415, 510],
        format_func=friendly_area_label,
        help="Choose the area code that matches the customer’s region."
    )
    international_plan = st.radio(
        "Does the customer use an international calling plan?",
        options=[0, 1],
        index=0,
        horizontal=True,
        format_func=friendly_yes_no,
        help="Choose Yes if the customer has an international plan."
    )
    voice_mail_plan = st.radio(
        "Does the customer use voicemail?",
        options=[0, 1],
        index=0,
        horizontal=True,
        format_func=friendly_yes_no,
        help="Choose Yes if the customer has a voicemail plan."
    )
    total_day_calls = st.number_input(
        "How many daytime calls did the customer make?",
        min_value=0,
        value=100,
        step=1,
        help="Enter the total number of daytime calls."
    )
    total_day_charge = st.number_input(
        "Daytime call charge",
        min_value=0.0,
        value=30.0,
        step=0.1,
        help="Enter the total charge for daytime calls."
    )
    total_eve_calls = st.number_input(
        "How many evening calls did the customer make?",
        min_value=0,
        value=100,
        step=1,
        help="Enter the total number of evening calls."
    )
    total_eve_charge = st.number_input(
        "Evening call charge",
        min_value=0.0,
        value=17.0,
        step=0.1,
        help="Enter the total charge for evening calls."
    )
    total_night_calls = st.number_input(
        "How many night calls did the customer make?",
        min_value=0,
        value=100,
        step=1,
        help="Enter the total number of night calls."
    )
    total_night_charge = st.number_input(
        "Night call charge",
        min_value=0.0,
        value=9.0,
        step=0.1,
        help="Enter the total charge for night calls."
    )
    total_intl_calls = st.number_input(
        "How many international calls did the customer make?",
        min_value=0,
        value=4,
        step=1,
        help="Enter the total number of international calls."
    )
    total_intl_charge = st.number_input(
        "International call charge",
        min_value=0.0,
        value=2.7,
        step=0.1,
        help="Enter the total charge for international calls."
    )
    number_customer_service_calls = st.number_input(
        "How many support calls did the customer make?",
        min_value=0,
        value=1,
        step=1,
        help="Enter the number of times the customer contacted support."
    )

    st.markdown("---")
    st.header("ℹ️ How it works")
    st.write("This tool helps you quickly estimate whether a customer may leave.")
    st.write("- Enter the customer details")
    st.write("- Click the prediction button")
    st.write("- Review the risk result and download a report")

input_data = pd.DataFrame([[
    account_length,
    area_code,
    international_plan,
    voice_mail_plan,
    total_day_calls,
    total_day_charge,
    total_eve_calls,
    total_eve_charge,
    total_night_calls,
    total_night_charge,
    total_intl_calls,
    total_intl_charge,
    number_customer_service_calls
]], columns=[
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
])

def format_feature_name(name):
    return name.replace("_", " ").title()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🔍 Predict Churn")
    if st.button("Run Prediction"):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if probability >= 0.66:
            risk_label = "High Risk"
            risk_color = "#c9302c"
        elif probability >= 0.33:
            risk_label = "Medium Risk"
            risk_color = "#f0ad4e"
        else:
            risk_label = "Low Risk"
            risk_color = "#198754"

        result_text = "Customer is likely to CHURN." if prediction == 1 else "Customer is NOT likely to churn."
        result_color = "#ffdddd" if prediction == 1 else "#ddffdd"
        text_color = "#900000" if prediction == 1 else "#006600"
        icon = "⚠️" if prediction == 1 else "✅"

        st.markdown(
            f"<div class='result-card' style='background:{result_color}; color:{text_color};'>"
            f"<div style='font-size:18px; font-weight:700;'>{icon} {result_text}</div>"
            f"<div class='big-probability'>{probability:.2%}</div>"
            f"<div class='risk-pill' style='background:{risk_color};'>{risk_label}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.progress(int(probability * 100))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_df = input_data.copy()
        output_df["prediction"] = prediction
        output_df["churn_probability"] = probability
        output_df["timestamp"] = timestamp

        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Report",
            data=csv,
            file_name="churn_prediction.csv",
            mime="text/csv"
        )
        st.write("*Includes customer inputs, prediction, probability, and timestamp.*")

        importance = None
        feature_names = input_data.columns.tolist()

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim == 2:
                coef = coef[0]
            importance = np.abs(coef)

        if importance is not None and len(importance) == len(feature_names):
            feature_display = [format_feature_name(name) for name in feature_names]
            fi_df = pd.DataFrame({
                "feature": feature_display,
                "importance": importance
            }).sort_values("importance", ascending=True)

            st.subheader("📊 Top Factors Affecting Prediction")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(fi_df["feature"], fi_df["importance"], color="#0d6efd")
            ax.set_xlabel("Importance")
            ax.set_ylabel("")
            ax.set_title("Feature Importance")
            ax.grid(axis="x", alpha=0.3)
            for i, v in enumerate(fi_df["importance"]):
                ax.text(v + max(fi_df["importance"]) * 0.01, i, f"{v:.2f}", va="center", color="#333")
            st.pyplot(fig)
        else:
            st.info("Feature importance is not available for this model type.")

with col_right:
    st.subheader("📝 Customer Information")
    info_df = pd.DataFrame({
        "Feature": [format_feature_name(c) for c in input_data.columns],
        "Value": input_data.iloc[0].values
    })
    st.table(info_df)
    st.markdown("---")
    st.subheader("📥 Download Report")
    st.write("Download the customer input and prediction results as a CSV after prediction.")
    st.write("*Includes customer inputs, prediction, probability, and timestamp.*")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6c757d; font-size:14px; margin-top:12px;'>"
    "📡 Telecom Customer Churn Prediction System — Built with <strong>Python</strong>, "
    "<strong>Scikit-learn</strong>, and <strong>Streamlit</strong> — Developed by "
    "<strong>Shayan</strong> — "
    "<a href='https://github.com/shayan-codes-405/Telco-Customer-Churn.git' style='color:#0d6efd; text-decoration:none;' target='_blank'>GitHub Repository</a>"
    "</div>",
    unsafe_allow_html=True
)