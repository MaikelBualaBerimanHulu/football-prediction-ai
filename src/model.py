"""
Model Training Module for Football Prediction System.

This module handles the training, evaluation, and serialization of the
Random Forest classifier used for match outcome prediction. It uses
enriched match data containing engineered features such as rolling form
scores and head-to-head statistics.

Author: [Your Name]
Date: 2026-06-15
Version: 2.1.0 (Restored missing train_and_evaluate function)
"""

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Constants for feature configuration
FEATURE_COLUMNS: List[str] = [
    "home_team_encoded",
    "away_team_encoded",
    "home_goals",
    "away_goals",
    "home_form_score",
    "away_form_score",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
]

TARGET_COLUMN: str = "result"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
N_ESTIMATORS: int = 100


def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Load enriched dataset and encode categorical team columns.

    Args:
        data_path: Path to the enriched CSV file.

    Returns:
        A tuple of (prepared DataFrame, dictionary of fitted LabelEncoders).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"[INFO] Loading enriched data from {data_path}...")
    df = pd.read_csv(data_path)

    # Validate RAW columns that should exist in the CSV
    raw_required_columns = {
        "home_team", "away_team", TARGET_COLUMN,
        "home_goals", "away_goals",
        "home_form_score", "away_form_score",
        "h2h_home_wins", "h2h_draws", "h2h_away_wins"
    }
    
    missing = raw_required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required raw columns in dataset: {missing}")

    # Encode team names to integers
    encoders: Dict[str, LabelEncoder] = {
        "home": LabelEncoder(),
        "away": LabelEncoder(),
    }

    df["home_team_encoded"] = encoders["home"].fit_transform(df["home_team"])
    df["away_team_encoded"] = encoders["away"].fit_transform(df["away_team"])

    print(f"[INFO] Data prepared: {len(df)} samples, {len(FEATURE_COLUMNS)} features")
    return df, encoders


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier and evaluate its performance.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        X_test: Test feature matrix.
        y_test: Test target labels.

    Returns:
        The trained RandomForestClassifier instance.
    """
    print("[INFO] Initializing Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all available CPU cores
    )

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n[RESULT] Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred))

    # Log feature importance
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n[INFO] Feature Importance Ranking:")
    for rank, (feat, imp) in enumerate(sorted_importance, 1):
        print(f"  {rank}. {feat}: {imp:.4f}")

    return model


def save_artifacts(
    model: RandomForestClassifier,
    encoders: Dict[str, LabelEncoder],
    models_dir: str,
) -> None:
    """
    Serialize trained model and encoders to disk.
    """
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "football_model_v2.pkl")
    encoder_path = os.path.join(models_dir, "encoders_v2.pkl")

    joblib.dump(model, model_path)
    joblib.dump(encoders, encoder_path)

    print(f"\n[INFO] Model saved to {model_path}")
    print(f"[INFO] Encoders saved to {encoder_path}")


def run_training_pipeline() -> None:
    """
    Execute the complete training pipeline end-to-end.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "matches_enriched.csv")
    models_dir = os.path.join(base_dir, "models")

    try:
        # Step 1: Load and prepare data
        df, encoders = load_and_prepare_data(data_path)

        # Step 2: Split into train/test sets
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        print(f"[INFO] Train set: {len(X_train)} samples")
        print(f"[INFO] Test set:  {len(X_test)} samples")

        # Step 3: Train and evaluate
        model = train_and_evaluate(X_train, y_train, X_test, y_test)

        # Step 4: Save artifacts
        save_artifacts(model, encoders, models_dir)

        print("\n[SUCCESS] Training pipeline completed successfully.")

    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_training_pipeline()