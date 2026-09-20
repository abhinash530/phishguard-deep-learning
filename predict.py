"""
PhishGuard - URL Prediction

Loads a trained PhishGuard model and predicts whether
a URL is potentially phishing or likely legitimate.
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from phishguard import extract_features


MODEL_PATH = "models/phishguard_model.keras"


def predict_url(url, model, scaler):
    """Predict the classification of a URL."""

    features = pd.DataFrame(
        [extract_features(url)]
    )

    features_scaled = scaler.transform(features)

    probability = float(
        model.predict(
            features_scaled,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        prediction = "Potential Phishing URL"
    else:
        prediction = "Likely Legitimate URL"

    return prediction, probability


def main():

    if not os.path.exists(MODEL_PATH):
        print("Trained model not found.")
        print(
            "Train the model first using "
            "train_model.py."
        )
        return

    model = load_model(MODEL_PATH)

    # The scaler should be fitted using the same
    # training data used by the model.
    print(
        "Model loaded successfully."
    )

    url = input(
        "\nEnter a URL to analyze: "
    ).strip()

    if not url:
        print("No URL entered.")
        return

    # For a production implementation, load the scaler
    # saved during training instead of fitting it here.
    print(
        "\nURL received for analysis:"
    )
    print(url)

    print(
        "\nNote: The prediction pipeline requires "
        "the training scaler to be saved and loaded "
        "alongside the model."
    )


if __name__ == "__main__":
    main()
