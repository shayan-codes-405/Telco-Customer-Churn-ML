# 📡 Telecom Customer Churn Prediction

A Machine Learning web application that predicts whether a telecom customer is likely to churn based on customer account details, service usage, and support history. The application is built with **Python**, **Scikit-learn**, **Streamlit**, and delivers fast, interpretable churn predictions with probability scores, risk labels, and downloadable reports.

---

## 🚀 Live Demo

The app is deployed to Streamlit Cloud and available at:

https://telco-customer-churn-mlgit-jxoyqjrh4fnrwdyps8q9fj.streamlit.app/

---

## 📌 Project Summary

This project helps telecom businesses identify customers who are at high risk of leaving the service. By predicting churn early, decision-makers can design retention campaigns, improve customer satisfaction, and preserve revenue.

### Business Value

- Reduce customer churn and improve customer lifetime value
- Identify at-risk customers for targeted retention offers
- Use model-driven insights to prioritize customer support
- Convert churn risk scores into operational actions for marketing and sales teams

---

## 📊 What Is Included

- Interactive Streamlit interface for real-time churn prediction
- Detailed customer input form with 13 service usage features
- Churn probability, risk label, and visual feedback
- Downloadable CSV report of prediction results
- Feature importance visualization
- Notebook with exploratory data analysis, model experiments, and evaluation graphs

---

## 🧠 Dataset Summary

The dataset includes telecom customer records with usage patterns, billing data, subscription features, and customer care activity.

- Dataset file: `Data_Churn.csv`
- Test file for prediction: `Data_Test.csv`
- Total records: 4,250
- Total features: 20
- No missing values and no duplicate records

### Feature List

| Feature | Description |
|---|---|
| state | Customer U.S. state |
| account_length | Months with the provider |
| area_code | Customer area code |
| international_plan | Has international calling plan |
| voice_mail_plan | Has voicemail plan |
| number_vmail_messages | Number of voicemail messages |
| total_day_minutes | Daytime call minutes |
| total_day_calls | Daytime call count |
| total_day_charge | Daytime call charges |
| total_eve_minutes | Evening call minutes |
| total_eve_calls | Evening call count |
| total_eve_charge | Evening call charges |
| total_night_minutes | Night call minutes |
| total_night_calls | Night call count |
| total_night_charge | Night call charges |
| total_intl_minutes | International call minutes |
| total_intl_calls | International call count |
| total_intl_charge | International call charges |
| number_customer_service_calls | Customer service calls count |
| churn | Target label: customer churned or not |

---

## 📈 Exploratory Data Analysis & Graphs

The notebook includes detailed analysis with the following visualizations and insights:

- Churn class distribution chart showing the percentage of churned vs non-churned customers
- Numerical feature distributions to inspect skewness and value ranges
- Bivariate KDE plots comparing feature distributions by churn status
- Categorical churn analysis for `area_code`, `international_plan`, and `voice_mail_plan`
- Correlation heatmap for feature relationships
- Pairplot and multivariate relationship analysis for international call metrics
- Boxplots to inspect outliers across numerical features

### Key EDA Takeaways

- Most customers do not churn, but churn rate spikes for customers with an international plan
- Customers with a voicemail plan also show a higher churn percent compared to those without
- Several call and charge features exhibit skewed distributions and should be standardized before training
- The dataset is clean and ready for modeling after encoding and scaling

---

## 🔧 Machine Learning Pipeline

### Preprocessing

- Categorical features are label-encoded
- Redundant features removed during feature selection
- Numeric features standardized using `StandardScaler`

### Modeling

The project evaluates multiple models:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Evaluation

- Cross-validation recall used during model selection
- ROC-AUC and confusion matrix used for model performance analysis
- Feature importance and SHAP analysis provide explainability

### Final Model

The final deployed model is a **Random Forest Classifier** saved as `model.pkl` and `scaler.pkl` for inference in the Streamlit app.

---

## 🚀 Streamlit App Features

- User-friendly sidebar form for customer input
- Real-time churn prediction with probability score
- Clear risk label: High / Medium / Low risk
- Download prediction report as CSV
- In-app feature importance chart
- Customer input summary table

---

## 📂 Project Structure

```text
Telco-Customer-Churn/
├── app.py
├── Predict_Churn.ipynb
├── Data_Churn.csv
├── Data_Test.csv
├── model.pkl
├── scaler.pkl
├── submit.csv
├── final.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/shayan-codes-405/Telco-Customer-Churn.git
```

Change directory:

```bash
cd Telco-Customer-Churn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📌 Notes

- The notebook contains full exploratory analysis, feature engineering, model training, and evaluation.
- `submit.csv` and `final.csv` are generated outputs for prediction results.
- The live demo is hosted on Streamlit Cloud.

---

## 👨‍💻 Author

**Shayan**

GitHub: https://github.com/shayan-codes-405

---

## 📜 License

This project is intended for educational and portfolio purposes.