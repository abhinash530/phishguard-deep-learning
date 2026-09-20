"""
PhishGuard - Dataset Loader

Loads a CSV dataset containing URLs and labels,
performs basic validation, and prepares the data
for feature extraction.
"""

import pandas as pd


def load_dataset(file_path):
    """
    Load the phishing dataset from a CSV file.

    Expected columns:
        url   -> URL string
        label -> 0 for legitimate, 1 for phishing
    """

    data = pd.read_csv(file_path)

    required_columns = {"url", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain 'url' and 'label' columns."
        )

    data = data.dropna(subset=["url", "label"])

    data["label"] = data["label"].astype(int)

    return data


def show_dataset_summary(data):
    """Display basic information about the dataset."""

    print("\n========== Dataset Summary ==========")
    print(f"Total records: {len(data)}")

    print("\nClass distribution:")
    print(data["label"].value_counts())

    print("\nMissing values:")
    print(data.isnull().sum())


if __name__ == "__main__":
    # Example:
    # data = load_dataset("dataset/phishing_dataset.csv")
    # show_dataset_summary(data)

    print("PhishGuard dataset loader is ready.")
    print("Place a properly sourced CSV dataset in the dataset directory.")
