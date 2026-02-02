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

GreenCart Europe is experiencing a high customer churn rate, with approximately 20% of customers stopping their purchases after a certain period. This negatively impacts revenue and long-term growth, as acquiring new customers is significantly more expensive than retaining existing ones.

The company wants to identify customers who are likely to churn in advance, so the marketing team can take proactive actions such as personalized promotions, retention campaigns, or subscription incentives.

A customer is considered churned if they stop making purchases for a defined period, as specified in the dataset.

Solving this problem will directly support GreenCart’s goal of increasing customer lifetime value and reducing marketing acquisition costs.

### Business Metrics
The main business metric impacted by this project is the customer churn rate.

| Metric | Now | Goal |
|--------|-----|------|
| Churn Rate | ~20% | 10% |
| Customer Lifetime Value | baseline | +25% |

Reducing churn is highly valuable, as retaining an existing customer is estimated to be 5–7 times cheaper than acquiring a new one.

### ML Approach

- **Learning Type:** Supervised learning
- **Problem Type:** Binary classification
- **Target Variable:** Churn (1 = churned, 0 = stayed)
- **Features:** Demographics, purchase behavior, website engagement,and satisfaction

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

The baseline gets ~80% accuracy just by always predicting "no churn" (since 80% of customers don't churn). But it's useless for actually finding churners - 0 recall means it never identifies anyone who will leave. This high accuracy is misleading, as it reflects the underlying class imbalance rather than predictive power. A ROC-AUC of 0.50 confirms that the model performs no better than random guessing.

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
