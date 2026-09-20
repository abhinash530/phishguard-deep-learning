from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

print("Downloading PhiUSIIL phishing URL dataset...")

dataset = fetch_ucirepo(id=967)

urls = dataset.data.features["URL"]
raw_labels = dataset.data.targets.iloc[:, 0].astype(int)

# UCI labels:
# 1 = legitimate
# 0 = phishing
#
# Our project uses:
# 0 = legitimate
# 1 = phishing

labels = 1 - raw_labels

df = pd.DataFrame({
    "url": urls,
    "label": labels
})

os.makedirs("dataset", exist_ok=True)

output_file = "dataset/phishing_dataset.csv"
df.to_csv(output_file, index=False)

print("\nDataset created successfully!")
print(f"File: {output_file}")
print(f"Total URLs: {len(df)}")

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nFirst 5 rows:")
print(df.head())
