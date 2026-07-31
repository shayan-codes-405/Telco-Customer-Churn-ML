# 📡 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

A Machine Learning web application that predicts whether a telecom customer is likely to churn based on customer account details, service usage, and support history. The application is built with **Python**, **Scikit-learn**, **Streamlit**, and delivers fast, interpretable churn predictions with probability scores, risk labels, and downloadable reports.

---
## 🚀 Live Demo

Experience the deployed application here:

🌐 **https://telco-customer-churn-mlgit-jxoyqjrh4fnrwdyps8q9fj.streamlit.app/**
---
<!-- 
## � Repository

Source code and project files are hosted on GitHub:

https://github.com/shayan-codes-405/Telco-Customer-Churn-ML -->



## �📌 Project Summary

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
## 📈 Exploratory Data Analysis (EDA)

Before building the machine learning model, exploratory data analysis was performed to understand customer behavior, identify feature distributions, and discover factors associated with customer churn.

---

### Customer Churn Distribution

The following chart shows the percentage of churned and non-churned customers in the dataset.

<p align="center">
  <img src="images/Percntage%20of%20churn.png" width="500">
</p>

**Key Takeaways:**

- The dataset is imbalanced, with the majority of customers not churning.
- Approximately **14%** of customers have churned.
- This distribution was considered while selecting evaluation metrics during model training.

---

### Univariate Analysis

Univariate analysis was performed to examine the distribution of each numerical feature individually.

<p align="center">
  <img src="images/univarate%20analysis.png" width="900">
</p>

**Key Takeaways:**

- Overall, the data appears to be **approximately normally distributed**.
- The features **number_vmail_messages**, **total_intl_calls**, and **number_customer_service_calls** exhibit **positively skewed distributions**.

---

### Bivariate Analysis

This analysis compares the distribution of each numerical feature between churned and non-churned customers to identify features related to customer churn.

<p align="center">
  <img src="images/bivarate%20anal.png" width="900">
</p>

---

### Categorical Feature Analysis

To avoid writing repetitive code, two helper functions were created.

- **ratio_with_target()** calculates the churn percentage for each category.
- **visualization()** displays both customer counts and churn percentages for a selected feature.

#### Area Code Analysis

<p align="center">
  <img src="images/catagrical%20analysis.png" width="650">
</p>

**Key Takeaways:**

- The majority of customers belong to **Area Code 415**.
- Customer churn remains consistent across all area codes, ranging between **14% and 15%**.

---

#### International Plan Analysis

<p align="center">
  <img src="images/catagrical%20analysis%202.png" width="650">
</p>

**Key Takeaways:**

- Although relatively few customers subscribe to an **International Plan**, **42%** of them churned.
- Customers **without** an International Plan have a churn rate of only **11%**.

---

#### Voice Mail Plan Analysis

<p align="center">
  <img src="images/catagrical%20analysis%203.png" width="650">
</p>

**Key Takeaways:**

- Customers with a **Voice Mail Plan** exhibit a churn rate of **16%**.
- Customers without a Voice Mail Plan show a comparatively lower churn rate.

---

### Summary of EDA

The exploratory analysis revealed several important business insights:

- Most customers do not churn, resulting in a moderately imbalanced dataset.
- Customers with an **International Plan** are significantly more likely to churn.
- Customers subscribed to a **Voice Mail Plan** also exhibit a relatively higher churn rate.
- Most numerical features are approximately normally distributed, while a few display positive skewness.
- These findings helped guide feature selection, preprocessing, and model development.
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
git clone https://github.com/shayan-codes-405/Telco-Customer-Churn-ML.git
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

GitHub repository: https://github.com/shayan-codes-405/Telco-Customer-Churn-ML

---
