# ChurnSense AI — Customer Churn Analysis & Prediction Dashboard 📊🤖

A modern Streamlit-based dashboard for customer churn analysis, visualization, and machine learning prediction.

---

## Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Understanding why customers leave and identifying at-risk customers can help organizations improve customer retention and reduce revenue loss.

**ChurnSense AI** is an interactive analytics dashboard that enables users to:

- Analyze customer churn behavior
- Visualize churn trends and patterns
- Identify high-risk customers
- Predict churn using Machine Learning
- Download churn analysis reports

---

## Business Problem

Acquiring new customers is often more expensive than retaining existing ones.

The objective of this project is to:

- Understand customer churn patterns
- Identify factors influencing churn
- Predict customers likely to leave
- Support data-driven retention strategies
- Improve business decision-making

---

## Features ✨

### 📊 Overview Dashboard
- Dataset preview
- Summary statistics
- Missing value analysis
- Key business metrics

### 📈 Analytics Dashboard
- Churn distribution analysis
- Customer tenure analysis
- Contract type impact analysis
- Correlation heatmap
- High-risk customer identification

### 🤖 Churn Prediction
- Machine Learning-based churn prediction
- Individual customer prediction form
- Interactive user inputs
- Real-time prediction results

### 📥 Reporting
- Download complete churn report
- Export high-risk customer list
- Generate business insights for decision-making

---

## Dashboard Screenshots

### Overview Page

![Overview Dashboard](assets/overview.png)

### Analytics Dashboard

![Analytics Dashboard](assets/analytics.png)

### Prediction Module

![Prediction Module](assets/prediction.png)

---

## Technology Stack 🛠️

| Technology | Purpose |
|------------|----------|
| Python | Core Programming Language |
| Streamlit | Interactive Dashboard Development |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Plotly | Interactive Visualizations |
| Scikit-Learn | Machine Learning |
| Random Forest Classifier | Churn Prediction Model |

---

## Dataset Information

The project uses the Telco Customer Churn Dataset.

### Important Features

- Customer ID
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn (Target Variable)

### Target Variable

| Value | Meaning |
|---------|---------|
| Yes | Customer Churned |
| No | Customer Retained |

---

## Data Preprocessing

The following preprocessing steps were performed:

- Missing value handling
- Data type correction
- Label Encoding for categorical features
- Numeric value conversion
- Feature selection
- Train-test splitting

---

## Machine Learning Model

### Model Used

**Random Forest Classifier**

Reasons for selection:

- Handles categorical and numerical data effectively
- Reduces overfitting through ensemble learning
- Provides strong baseline performance
- Suitable for churn classification problems

---

## Key Insights

The analysis revealed several important patterns:

- Customers with month-to-month contracts showed higher churn rates.
- Customers with shorter tenure were more likely to leave.
- Electronic check payment users demonstrated increased churn behavior.
- Long-term contract customers exhibited better retention.
- Monthly charges showed a relationship with customer churn.

---

## 📈 Model Performance

| Metric | Score |
|----------|---------|
| Accuracy | 81.33%|
| Precision | 67.97% |
| Recall | 55.76% |
| F1 Score | 61.27%|

---

## Project Workflow

```text
Dataset Upload
       ↓
Data Cleaning & Preprocessing
       ↓
Exploratory Data Analysis (EDA)
       ↓
Model Training
       ↓
Churn Prediction
       ↓
Business Insights & Reporting
```

---

## Quick Start 🚀

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

---

## Future Enhancements 🚀

Planned improvements include:

- XGBoost and LightGBM models
- Hyperparameter tuning
- Customer churn risk scoring
- Automated PDF report generation
- Model persistence using Joblib
- Cloud deployment
- Retention recommendation system
- Advanced business intelligence dashboard

---

## Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning
- Customer Segmentation Analysis
- Dashboard Development
- Business Intelligence Reporting
- Streamlit Application Development

## Project Structure

```text
churn-dashboard/
│
├── app.py
├── model.py
├── requirements.txt
├── assets/
│   ├── overview.png
│   ├── analytics.png
│   └── prediction.png
│
└── data/
    └── Telco_Customer_Churn.csv
```

---

## Conclusion

ChurnSense AI demonstrates how data analytics and machine learning can be combined to understand customer behavior, identify churn risks, and support business retention strategies.

The dashboard provides an interactive and user-friendly environment for analyzing customer churn data and generating actionable insights for decision-makers.

---

## Developed By

**Helani Ambalangodage**

Customer Churn Analysis & Prediction Project

**Streamlit | Python | Machine Learning | Data Analytics**