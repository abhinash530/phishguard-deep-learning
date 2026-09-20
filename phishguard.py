"""
PhishGuard - URL Feature Extraction

Provides reusable functions for extracting lexical
features from URLs for phishing detection.
"""

import re
import pandas as pd


def extract_features(url):
    """
    Extract lexical features from a URL.

    Returns:
        Dictionary containing numerical URL features.
    """

    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_at": url.count("@"),
        "num_digits": sum(
            character.isdigit()
            for character in url
        ),
        "num_special_chars": len(
            re.findall(
                r"[^a-zA-Z0-9]",
                url
            )
        ),
        "https": int(
            url.lower().startswith("https")
        ),
        "has_ip_address": int(
            bool(
                re.search(
                    r"(?:\d{1,3}\.){3}\d{1,3}",
                    url
                )
            )
        )
    }

    return features


def extract_features_from_dataframe(data):
    """
    Convert a dataframe containing URLs into
    a numerical feature dataframe.
    """

    return pd.DataFrame(
        [
            extract_features(url)
            for url in data["url"]
        ]
    )


if __name__ == "__main__":

    # Simple feature extraction demonstration.
    example_url = (
        "http://secure-login-example.com/verify"
    )

    features = extract_features(
        example_url
    )

    print("\nPhishGuard Feature Extraction")
    print("=" * 35)

    for feature, value in features.items():
        print(
            f"{feature}: {value}"
        )
