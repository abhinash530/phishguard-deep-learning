"""
PhishGuard - Model Training

Trains a neural-network classifier using URL features
generated from a CSV phishing dataset.
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from phishguard import extract_features
from model_utils import save_scaler


DATASET_PATH = "dataset/phishing_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "phishguard_model.keras"
)


def prepare_features(data):
    """Convert URLs into numerical feature vectors."""

    features = pd.DataFrame(
        [extract_features(url) for url in data["url"]]
    )

    labels = data["label"].astype(int)

    return features, labels


def build_model(input_size):
    """Create the PhishGuard neural-network model."""

    model = Sequential([
        Dense(
            64,
            activation="relu",
            input_shape=(input_size,)
        ),

        Dropout(0.30),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.20),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    # Check whether the dataset exists
    if not os.path.exists(DATASET_PATH):

        print("\nDataset not found.")
        print(
            f"Expected file: {DATASET_PATH}"
        )

        print(
            "\nAdd a properly sourced CSV dataset "
            "with the following columns:"
        )

        print("url,label")

        return

    # Load dataset
    data = pd.read_csv(
        DATASET_PATH
    )

    # Validate columns
    required_columns = {
        "url",
        "label"
    }

    if not required_columns.issubset(
        data.columns
    ):
        raise ValueError(
            "Dataset must contain "
            "'url' and 'label' columns."
        )

    # Remove incomplete records
    data = data.dropna(
        subset=["url", "label"]
    )

    # Convert labels to integers
    data["label"] = data["label"].astype(int)

    print("\n========== Dataset ==========")
    print(
        f"Total records: {len(data)}"
    )

    print("\nClass distribution:")
    print(
        data["label"].value_counts()
    )

    # Generate numerical features
    X, y = prepare_features(
        data
    )

    # Train-test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # Save fitted scaler
    save_scaler(
        scaler
    )

    # Build model
    model = build_model(
        X_train.shape[1]
    )

    print(
        "\n========== Training =========="
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.20,
        verbose=1
    )

    # Generate predictions
    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int).flatten()

    # Evaluation metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        "\n========== Evaluation =========="
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Phishing"
            ],
            zero_division=0
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # Create model directory
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save trained model
    model.save(
        MODEL_PATH
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        "Training completed successfully."
    )


if __name__ == "__main__":
    main()
