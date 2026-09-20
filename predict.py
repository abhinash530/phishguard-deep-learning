"""
PhishGuard - URL Prediction

Loads the trained PhishGuard model and the fitted feature
scaler, then predicts whether a URL is potentially phishing.
"""

import os
import pandas as pd
from tensorflow.keras.models import load_model

from phishguard import extract_features
from model_utils import load_scaler


MODEL_PATH = "models/phishguard_model.keras"


def predict_url(url, model, scaler):
    """Predict whether a URL is legitimate or phishing."""

    features = pd.DataFrame(
        [extract_features(url)]
    )

    features_scaled = scaler.transform(
        features
    )

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
        print("\nTrained model not found.")
        print(
            "Please train the model first using "
            "train_model.py."
        )
        return

    try:
        model = load_model(
            MODEL_PATH
        )

        scaler = load_scaler()

    except Exception as error:
        print(
            f"\nError loading model or scaler: {error}"
        )
        return

    print("\n================================")
    print("       PhishGuard Scanner")
    print("================================")

    url = input(
        "\nEnter a URL to analyze: "
    ).strip()

    if not url:
        print("No URL entered.")
        return

    prediction, probability = predict_url(
        url,
        model,
        scaler
    )

    print("\n========== Result ==========")
    print(f"URL: {url}")
    print(f"Prediction: {prediction}")
    print(
        f"Phishing Probability: "
        f"{probability:.2%}"
    )


if __name__ == "__main__":
    main()
