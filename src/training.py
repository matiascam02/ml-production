# src/training.py
# Training pipeline for Telco Customer Churn prediction
# Reads data from GCS, trains model, saves artifacts to GCS

import io
import argparse
import warnings
import pandas as pd
import numpy as np
import joblib
from google.cloud import storage
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

warnings.filterwarnings('ignore', category=FutureWarning)

# GCS config
BUCKET_NAME = 'greencart-churn-regression'
PROJECT_ID = 'mlip-485615'

# Feature definitions
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


def load_data_from_gcs(bucket, blob_path):
    """Download CSV from GCS and return as DataFrame."""
    blob = bucket.blob(blob_path)
    csv_bytes = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from gs://{BUCKET_NAME}/{blob_path}")
    return df


def clean_data(df):
    """Clean the Telco dataset: fix types, encode target, drop ID."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    if not pd.api.types.is_numeric_dtype(df['Churn']):
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    print(f"Cleaned: {df.shape[0]} rows, churn rate: {df['Churn'].mean()*100:.1f}%")
    return df


def split_data(df, test_size=0.20, val_size=0.25, seed=42):
    """Split into 60/20/20 train/val/test."""
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df['Churn']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=y_temp
    )

    for name, xs, ys in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
        print(f"  {name:5}: {len(xs):5} samples ({ys.mean()*100:.1f}% churn)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    """Build ColumnTransformer for numerical + categorical features."""
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])


def evaluate(model, X, y):
    """Compute classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1-Score': f1_score(y, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y, y_proba),
    }


def train_logistic_regression(X_train, y_train):
    """Train LR with GridSearchCV."""
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
    }
    search = GridSearchCV(
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1:  {search.best_score_:.4f}")
    return search.best_estimator_


def train_random_forest(X_train, y_train):
    """Train RF with GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [5, 10, 20],
        'class_weight': ['balanced'],
    }
    search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1:  {search.best_score_:.4f}")
    return search.best_estimator_


def upload_artifact(bucket, obj, gcs_path):
    """Serialize object with joblib and upload to GCS."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(buffer)
    print(f"  Uploaded: gs://{BUCKET_NAME}/{gcs_path}")


def main():
    parser = argparse.ArgumentParser(description='Train churn prediction models')
    parser.add_argument('--input', default='inputs/telco_customer_churn.csv',
                        help='GCS path to input CSV (default: inputs/telco_customer_churn.csv)')
    parser.add_argument('--model', default='lr', choices=['lr', 'rf', 'both'],
                        help='Model to train: lr, rf, or both (default: lr)')
    args = parser.parse_args()

    # Connect to GCS
    print("Connecting to GCS...")
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    # Load and clean data
    print("\n--- Loading Data ---")
    df = load_data_from_gcs(bucket, args.input)
    df = clean_data(df)

    # Split
    print("\n--- Splitting Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Preprocess
    print("\n--- Preprocessing ---")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    print(f"  Features after encoding: {X_train_proc.shape[1]}")

    results = {}

    # Train Logistic Regression
    if args.model in ('lr', 'both'):
        print("\n--- Training Logistic Regression ---")
        lr_model = train_logistic_regression(X_train_proc, y_train)
        lr_val = evaluate(lr_model, X_val_proc, y_val)
        lr_test = evaluate(lr_model, X_test_proc, y_test)
        results['Logistic Regression'] = {'val': lr_val, 'test': lr_test}

        print("\n  Validation metrics:")
        for k, v in lr_val.items():
            print(f"    {k:12}: {v:.4f}")
        print("\n  Test metrics:")
        for k, v in lr_test.items():
            print(f"    {k:12}: {v:.4f}")

    # Train Random Forest
    if args.model in ('rf', 'both'):
        print("\n--- Training Random Forest ---")
        rf_model = train_random_forest(X_train_proc, y_train)
        rf_val = evaluate(rf_model, X_val_proc, y_val)
        rf_test = evaluate(rf_model, X_test_proc, y_test)
        results['Random Forest'] = {'val': rf_val, 'test': rf_test}

        print("\n  Validation metrics:")
        for k, v in rf_val.items():
            print(f"    {k:12}: {v:.4f}")
        print("\n  Test metrics:")
        for k, v in rf_test.items():
            print(f"    {k:12}: {v:.4f}")

    # Save artifacts to GCS
    print("\n--- Saving Artifacts to GCS ---")
    upload_artifact(bucket, preprocessor, 'artifacts/preprocessor.joblib')

    if args.model in ('lr', 'both'):
        upload_artifact(bucket, lr_model, 'artifacts/logistic_regression.joblib')
    if args.model in ('rf', 'both'):
        upload_artifact(bucket, rf_model, 'artifacts/random_forest.joblib')

    # Save metrics CSV to GCS
    rows = []
    for model_name, metrics in results.items():
        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                rows.append({'model': model_name, 'split': split_name,
                             'metric': metric_name, 'value': round(value, 4)})
    metrics_df = pd.DataFrame(rows)
    blob = bucket.blob('outputs/training_metrics.csv')
    blob.upload_from_string(metrics_df.to_csv(index=False), content_type='text/csv')
    print(f"  Uploaded: gs://{BUCKET_NAME}/outputs/training_metrics.csv")

    print("\nDone.")


if __name__ == '__main__':
    main()
