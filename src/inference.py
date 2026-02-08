# src/inference.py
# Inference pipeline for Telco Customer Churn prediction
# Loads trained model from GCS, runs predictions on new data, saves results to GCS

import io
import argparse
import warnings
import pandas as pd
import numpy as np
import joblib
from google.cloud import storage

warnings.filterwarnings('ignore', category=FutureWarning)

# GCS config
BUCKET_NAME = 'greencart-churn-regression'
PROJECT_ID = 'mlip-485615'

# Feature definitions (must match training)
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


def download_artifact(bucket, gcs_path):
    """Download a joblib artifact from GCS."""
    blob = bucket.blob(gcs_path)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    obj = joblib.load(buffer)
    print(f"  Loaded: gs://{BUCKET_NAME}/{gcs_path}")
    return obj


def load_data_from_gcs(bucket, blob_path):
    """Download CSV from GCS and return as DataFrame."""
    blob = bucket.blob(blob_path)
    csv_bytes = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns from gs://{BUCKET_NAME}/{blob_path}")
    return df


def clean_data(df):
    """Clean the Telco dataset for inference."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Keep customerID for output mapping if present
    customer_ids = df['customerID'] if 'customerID' in df.columns else None
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Encode Churn if present (for evaluation), otherwise ignore
    has_labels = 'Churn' in df.columns
    if has_labels and not pd.api.types.is_numeric_dtype(df['Churn']):
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    return df, customer_ids, has_labels


def main():
    parser = argparse.ArgumentParser(description='Run churn predictions on new data')
    parser.add_argument('--input', default='inputs/telco_customer_churn.csv',
                        help='GCS path to input CSV (default: inputs/telco_customer_churn.csv)')
    parser.add_argument('--model', default='lr', choices=['lr', 'rf'],
                        help='Which model to use: lr or rf (default: lr)')
    parser.add_argument('--output', default='outputs/predictions.csv',
                        help='GCS path for output predictions (default: outputs/predictions.csv)')
    args = parser.parse_args()

    model_paths = {
        'lr': 'artifacts/logistic_regression.joblib',
        'rf': 'artifacts/random_forest.joblib',
    }

    # Connect to GCS
    print("Connecting to GCS...")
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    # Load model and preprocessor
    print("\n--- Loading Artifacts ---")
    preprocessor = download_artifact(bucket, 'artifacts/preprocessor.joblib')
    model = download_artifact(bucket, model_paths[args.model])
    model_name = 'Logistic Regression' if args.model == 'lr' else 'Random Forest'
    print(f"  Using model: {model_name}")

    # Load and clean data
    print("\n--- Loading Data ---")
    df = load_data_from_gcs(bucket, args.input)
    df_clean, customer_ids, has_labels = clean_data(df)

    # Extract features and preprocess
    X = df_clean[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    X_processed = preprocessor.transform(X)
    print(f"  Preprocessed {X_processed.shape[0]} samples, {X_processed.shape[1]} features")

    # Predict
    print("\n--- Running Predictions ---")
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    # Build output DataFrame
    output = pd.DataFrame({
        'churn_prediction': predictions,
        'churn_probability': probabilities.round(4),
    })
    if customer_ids is not None:
        output.insert(0, 'customerID', customer_ids.values)

    # Print summary
    n_churners = predictions.sum()
    print(f"  Predicted churners: {n_churners} / {len(predictions)} ({n_churners/len(predictions)*100:.1f}%)")

    # If labels exist, print evaluation metrics
    if has_labels:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        y_true = df_clean['Churn']
        print(f"\n  Evaluation (labels available):")
        print(f"    Accuracy:  {accuracy_score(y_true, predictions):.4f}")
        print(f"    F1-Score:  {f1_score(y_true, predictions, zero_division=0):.4f}")
        print(f"    ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}")

    # Upload predictions to GCS
    print(f"\n--- Saving Predictions ---")
    blob = bucket.blob(args.output)
    blob.upload_from_string(output.to_csv(index=False), content_type='text/csv')
    print(f"  Uploaded: gs://{BUCKET_NAME}/{args.output}")

    print("\nDone.")


if __name__ == '__main__':
    main()
