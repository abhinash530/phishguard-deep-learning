# PhishGuard - Deep Learning Based Phishing URL Detection

PhishGuard is a cybersecurity machine learning project that detects potentially
phishing URLs using URL-based feature extraction and a deep learning
classification model.

The system analyzes characteristics of a URL such as its length, number of
dots, hyphens, slashes, digits, special characters, HTTPS usage, and whether
the URL contains an IP address.

---

## Project Objective

The objective of PhishGuard is to build a lightweight URL-based phishing
detection system that can classify URLs as:

- `0` - Legitimate
- `1` - Phishing

The project demonstrates the application of Python, machine learning,
deep learning, and cybersecurity concepts to phishing URL detection.

---

## Technologies Used

- Python 3.11
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib
- Git & GitHub

---

## Dataset

PhishGuard uses the **PhiUSIIL Phishing URL (Website) Dataset** from the
UCI Machine Learning Repository.

The dataset contains 235,795 URL records.

For this project, the original dataset labels were converted to the
following convention:

- `0` = Legitimate
- `1` = Phishing

The dataset is downloaded locally using `download_dataset.py` and is not
stored directly in this GitHub repository because of its size.

Dataset source:

https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

---

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
├── download_dataset.py
├── model_utils.py
├── phishguard.py
├── predict.py
├── requirements.txt
└── train_model.py
