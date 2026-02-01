# Customer Churn Prediction - GreenCart Europe

## A. Company Information

**Company Name:** GreenCart Europe

**Product/Service:** Online e-commerce platform selling eco-friendly household and lifestyle products.

**Business Model:** B2C model - revenue from direct product sales plus a premium subscription (discounts + free delivery).

**Size:** ~250 employees

**Mission:** Make eco-friendly products accessible and affordable across Europe.

**Sector:** E-commerce / Retail / Sustainability

**Location:** Berlin, Germany (HQ), operations across Europe

---

## B. Problem Statement

### The Problem

GreenCart is losing too many customers. About 20% of customers stop buying after a while, which hurts revenue since getting new customers is way more expensive than keeping existing ones.

We want to predict which customers are likely to churn so the marketing team can reach out with targeted offers before they leave.

### Business Metrics

| Metric | Now | Goal |
|--------|-----|------|
| Churn Rate | ~20% | 10% |
| Customer Lifetime Value | baseline | +25% |

Reducing churn saves money - keeping a customer costs 5-7x less than acquiring a new one.

### ML Approach

- **Type:** Supervised binary classification
- **Target:** Churn (1 = churned, 0 = stayed)
- **Features:** Demographics, purchase behavior, website engagement

### Evaluation Metrics

We'll track multiple metrics since this is an imbalanced problem:

- **F1-Score** (primary) - balances precision and recall
- **ROC-AUC** - overall model quality
- **Recall** - we don't want to miss churners
- **Precision** - avoid wasting resources on false alarms

### Dataset

Source: [Kaggle - Online Retail Customer Churn](https://www.kaggle.com/datasets/sahilislam007/online-retail-customer-churn-prediction-dataset)

- 9,000 customers
- 17 features (demographics, purchases, engagement, etc.)
- Split: 60% train / 20% validation / 20% test

---

## C. Exploratory Data Analysis

See `notebooks/exploratory_data_analysis.ipynb`

Key findings:
- ~20% churn rate (imbalanced classes, roughly 4:1)
- No missing values
- Mix of numerical and categorical features
- Some features show correlation with churn (need to verify in notebook)

---

## D. Baseline Model

The baseline just predicts the majority class (no churn) for everyone. It's the simplest possible model and gives us something to beat.

**Results:**

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 80.3% | 80.3% |
| Precision | 0.0 | 0.0 |
| Recall | 0.0 | 0.0 |
| F1-Score | 0.0 | 0.0 |
| ROC-AUC | 0.50 | 0.50 |

The baseline gets ~80% accuracy just by always predicting "no churn" (since 80% of customers don't churn). But it's useless for actually finding churners - 0 recall means it never identifies anyone who will leave.

Any real model needs to beat ROC-AUC 0.50 and actually detect some churners.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── baseline.py
│   └── utils.py
└── data/
    ├── online_retail_churn.csv
    ├── dataset_sample.csv
    └── baseline_metrics.csv
```

## Setup

```bash
pip install -r requirements.txt
```

Then run the notebook to download data and generate outputs.
