# PhishGuard – Phishing Detection Using Deep Learning

PhishGuard is an academic cybersecurity and deep learning project designed to analyze URLs and classify them as potentially phishing or likely legitimate.

The project demonstrates how URL-based lexical features can be extracted and processed using a neural-network classification model.

## Project Objective

The objective of PhishGuard is to develop a machine-learning-based approach for identifying potentially malicious URLs.

The project covers:

- URL feature extraction
- Dataset preprocessing
- Feature scaling
- Neural-network model training
- Model evaluation
- Phishing probability prediction

## Features

- Extracts lexical characteristics from URLs
- Detects potentially suspicious URL patterns
- Uses a neural-network classifier
- Supports training and evaluation using a CSV dataset
- Saves the trained model for later predictions
- Saves the feature scaler used during training
- Provides a command-line prediction interface

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

## Project Structure

```text
phishguard-deep-learning/
│
├── dataset/
│   └── README.md
│
├── .gitignore
├── README.md
├── data_loader.py
├── model_utils.py
├── phishguard.py
├── predict.py
├── requirements.txt
└── train_model.py
