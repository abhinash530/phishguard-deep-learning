"""
PhishGuard - Model Utilities

Utility functions for saving and loading the feature scaler
used by the PhishGuard training and prediction pipeline.
"""

import os
import joblib


MODEL_DIR = "models"
SCALER_PATH = os.path.join(
    MODEL_DIR,
    "feature_scaler.pkl"
)


def save_scaler(scaler):
    """
    Save the fitted feature scaler.

    Parameters:
        scaler: Fitted sklearn scaler.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print(
        f"Feature scaler saved to: {SCALER_PATH}"
    )


def load_scaler():
    """
    Load the previously saved feature scaler.

    Returns:
        Fitted scaler object.
    """

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Feature scaler not found. "
            "Train the model first."
        )

    return joblib.load(
        SCALER_PATH
    )
