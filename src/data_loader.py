import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATASET_PATH, TEST_SIZE, RANDOM_STATE


def load_dataset(dataset_path=DATASET_PATH):
    """
    Loads the European credit card fraud dataset.
    Features: Time, V1-V28, Amount (30 features total)
    Target: Class (0: legitimate, 1: fraud)
    """
    df = pd.read_csv(dataset_path)
    
    # Handle possible column name case inconsistencies
    if "Class" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "Class"})
    if "Amount" not in df.columns and "amount" in df.columns:
        df = df.rename(columns={"amount": "Amount"})
    if "Time" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "Time"})

    # Ensure class is integer binary
    df["Class"] = df["Class"].astype(int)
    return df


def prepare_train_test_split(df=None, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Performs stratified 70/30 train-test split and feature standardization.
    Preserves raw unscaled transaction amounts for cost-sensitive operations.
    """
    if df is None:
        df = load_dataset()

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()

    # Retain raw unscaled amounts for cost calculation
    amounts = df["Amount"].values.copy()

    # Stratified split maintaining class imbalance (70% train, 30% test)
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X,
        y,
        amounts,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Standardize all 30 features (Time, V1-V28, Amount)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "amounts_train": amounts_train,
        "amounts_test": amounts_test,
        "feature_names": feature_cols,
        "scaler": scaler,
    }
