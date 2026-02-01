# src/utils.py
# Helper functions for the churn project

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Features we use for modeling (excluding CustomerID and date columns)
FEATURE_COLS = [
    'Age', 'Gender', 'Annual_Income_USD', 'Spending_Score',
    'Membership_Status', 'Preferred_Payment_Method', 'Region',
    'Total_Purchases', 'Avg_Purchase_Value', 'Satisfaction_Score',
    'Website_Visits_Last_Month', 'Avg_Time_Per_Visit_Minutes',
    'Support_Tickets_Last_6_Months', 'Referred_Friends'
]


def load_data(file_path):
    """Load CSV and print basic info."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_data(df, target='Churn', test_size=0.2, val_size=0.25, seed=42):
    """
    Split data into train/val/test sets.
    Default split: 60% train, 20% val, 20% test
    """
    X = df[FEATURE_COLS].copy()
    y = df[target].copy()
    
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Then split train/val from the rest
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_info(X_train, X_val, X_test, y_train, y_val, y_test):
    """Show split sizes and churn rates."""
    total = len(X_train) + len(X_val) + len(X_test)
    
    print(f"Train:      {len(X_train):5} ({len(X_train)/total*100:.0f}%) - {y_train.mean()*100:.1f}% churn")
    print(f"Validation: {len(X_val):5} ({len(X_val)/total*100:.0f}%) - {y_val.mean()*100:.1f}% churn")
    print(f"Test:       {len(X_test):5} ({len(X_test)/total*100:.0f}%) - {y_test.mean()*100:.1f}% churn")
