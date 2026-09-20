"""
PhishGuard - Phishing Detection Using Deep Learning

Academic project for detecting phishing URLs using
URL-based lexical features and a neural network classifier.
"""

import re
import numpy as np
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


# ---------------------------------------------------------
# 1. Feature Extraction
# ---------------------------------------------------------

def extract_features(url):
    """Extract basic lexical features from a URL."""

    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_at": url.count("@"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9]", url)),
        "https": int(url.lower().startswith("https")),
        "has_ip_address": int(
            bool(re.search(r"(?:\d{1,3}\.){3}\d{1,3}", url))
        ),
    }

    return features


# ---------------------------------------------------------
# 2. Example Dataset
# ---------------------------------------------------------

# 0 = Legitimate
# 1 = Phishing
#
# These examples are included only to demonstrate the
# complete workflow. For a real experiment, replace this
# section with a properly sourced phishing dataset.

urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://www.amazon.com",
    "https://www.apple.com",
    "https://www.python.org",
    "https://www.linkedin.com",
    "http://secure-login-example.com",
    "http://account-verification-example.com/login",
    "http://192.168.1.20/login",
    "http://verify-user-example.com/account",
    "http://free-prize-example.com/winner",
    "http://update-payment-example.com/verify",
    "http://login-security-example.com/signin",
    "http://bank-account-example.com/confirm",
]

labels = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1
]


# ---------------------------------------------------------
# 3. Create Feature Dataset
# ---------------------------------------------------------

data = pd.DataFrame([extract_features(url) for url in urls])
data["label"] = labels

X = data.drop("label", axis=1)
y = data["label"]


# ---------------------------------------------------------
# 4. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 5. Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------
# 6. Build Deep Learning Model
# ---------------------------------------------------------

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.30),

    Dense(32, activation="relu"),
    Dropout(0.20),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


# ---------------------------------------------------------
# 7. Compile Model
# ---------------------------------------------------------

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------------------------------------
# 8. Train Model
# ---------------------------------------------------------

print("\nTraining PhishGuard model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)


# ---------------------------------------------------------
# 9. Model Evaluation
# ---------------------------------------------------------

probabilities = model.predict(X_test, verbose=0)

predictions = (probabilities >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n========== PhishGuard Evaluation ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ---------------------------------------------------------
# 10. Test a New URL
# ---------------------------------------------------------

def predict_url(url):
    """Predict whether a URL is legitimate or phishing."""

    features = pd.DataFrame([extract_features(url)])
    features_scaled = scaler.transform(features)

    probability = model.predict(
        features_scaled,
        verbose=0
    )[0][0]

    if probability >= 0.5:
        result = "Potential Phishing URL"
    else:
        result = "Likely Legitimate URL"

    print("\nURL:", url)
    print("Prediction:", result)
    print(f"Phishing Probability: {probability:.2%}")


# Example prediction
predict_url("http://secure-login-example.com/verify-account")
