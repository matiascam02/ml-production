# Customer Churn Prediction - EuroConnect Telecom

## A. Company Information

**Company Name:** EuroConnect Telecom

**Product/Service:** Telecommunications provider offering phone service, internet (DSL and fiber optic), and streaming packages to residential customers.

**Business Model:** B2C subscription model - revenue from monthly service plans (phone, internet, TV/streaming) with month-to-month, one-year, and two-year contract options.

**Size:** ~250 employees

**Mission:** Provide reliable and affordable telecom services across Europe.

**Sector:** Telecommunications

**Location:** Berlin, Germany (HQ), operations across Europe

---

## B. Problem Statement

### The Problem

EuroConnect Telecom is experiencing a high customer churn rate, with approximately 27% of subscribers cancelling their services. This negatively impacts revenue and long-term growth, as acquiring new subscribers is significantly more expensive than retaining existing ones.

The company wants to identify customers who are likely to churn in advance, so the retention team can take proactive actions such as personalized offers, contract upgrades, or service improvements.

A customer is considered churned if they cancel all services with the company.

Solving this problem will directly support EuroConnect's goal of increasing customer lifetime value and reducing acquisition costs.

### Business Metrics
The main business metric impacted by this project is the customer churn rate.

| Metric | Now | Goal |
|--------|-----|------|
| Churn Rate | ~27% | 15% |
| Customer Lifetime Value | baseline | +25% |

Reducing churn is highly valuable, as retaining an existing customer is estimated to be 5-7 times cheaper than acquiring a new one.

### ML Approach

- **Learning Type:** Supervised learning
- **Problem Type:** Binary classification
- **Target Variable:** Churn (1 = churned, 0 = stayed)
- **Features:** Demographics, account info, service subscriptions, billing

### Evaluation Metrics

Since customer churn is an imbalanced classification problem, multiple evaluation metrics are required:

- **F1-Score** (primary) - Balances precision and recall, ensuring that churners are identified while avoiding excessive false positives.
- **ROC-AUC** - Measures overall model discrimination performance.
- **Recall** - Important to minimize the number of churners that are missed.
- **Precision** - Helps control marketing costs by avoiding unnecessary retention efforts.

The choice of metrics reflects the trade-off between retention effectiveness and marketing efficiency.

### Dataset

Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers
- 20 features (demographics, account, services, billing)
- ~26.5% churn rate
- Data Split: 60% train / 20% validation / 20% test (stratified)
- The test dataset is kept untouched until the final evaluation, simulating real-world model deployment.

---

## C. Exploratory Data Analysis

See `notebooks/exploratory_data_analysis.ipynb`

Key findings:
- 26.5% churn rate (imbalanced classes, roughly 73:27)
- 11 blank values in TotalCharges (new customers with tenure=0), filled with 0
- Strongest churn predictors:
  - **Contract type**: month-to-month customers churn at 42.7% vs 2.8% for two-year contracts
  - **Tenure**: strong negative correlation (-0.35) — longer customers stay more
  - **Internet service**: fiber optic customers churn at 41.9% vs 18.9% DSL
  - **Monthly charges**: churners pay ~$15 more on average
- Gender and phone service show no significant difference in churn rates

---

## D. Baseline Model

The baseline predicts the majority class (no churn) for everyone.

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 73.5% | 73.5% |
| Precision | 0.0 | 0.0 |
| Recall | 0.0 | 0.0 |
| F1-Score | 0.0 | 0.0 |
| ROC-AUC | 0.50 | 0.50 |

The baseline gets ~73% accuracy just by always predicting "no churn", but it never identifies anyone who will leave (0 recall). Any real model needs to beat ROC-AUC 0.50 and actually detect churners.

---

## E. Model Training

See `notebooks/model_training.ipynb`

Two models were trained with GridSearchCV (5-fold CV, F1 scoring):

| Metric | Baseline | Logistic Regression | Random Forest |
|--------|----------|---------------------|---------------|
| Accuracy | 0.7346 | 0.7544 | **0.7601** |
| Precision | 0.0000 | 0.5248 | **0.5335** |
| Recall | 0.0000 | **0.7914** | 0.7674 |
| F1-Score | 0.0000 | **0.6311** | 0.6294 |
| ROC-AUC | 0.5000 | 0.8361 | **0.8381** |

Both models significantly outperform the baseline. Logistic Regression is the recommended model for its interpretability and slightly better recall/F1.

---

## F. Google Cloud Storage

- **Project:** `mlip-485615`
- **Bucket:** `gs://greencart-churn-regression` (europe-west3)

Bucket structure:
```
greencart-churn-regression/
├── inputs/              # Raw data
│   └── telco_customer_churn.csv
├── artifacts/           # Trained models and preprocessor
│   ├── preprocessor.joblib
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
└── outputs/             # Predictions and metrics
    ├── training_metrics.csv
    ├── model_comparison_metrics.csv
    └── predictions.csv
```

IAM roles — see `docs/iam_permissions.pdf` for details.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── docs/
│   └── iam_permissions.pdf
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   ├── baseline.py        # Majority-class baseline model
│   ├── utils.py            # Shared feature definitions and helpers
│   ├── training.py         # Training pipeline (reads/writes GCS)
│   └── inference.py        # Inference pipeline (reads/writes GCS)
└── data/
    └── telco_customer_churn.csv
```

## Setup

```bash
pip install -r requirements.txt
```

### Training
```bash
python src/training.py --model both
```

### Inference
```bash
python src/inference.py --model lr --input inputs/telco_customer_churn.csv
```
