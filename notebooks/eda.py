import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/creditcard.csv/creditcard.csv")

# -----------------------------
# Basic Information
# -----------------------------
print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\n")
print("=" * 50)
print("Dataset Information")
print(df.info())

print("\n")
print("=" * 50)
print("Missing Values")
print(df.isnull().sum())

print("\n")
print("=" * 50)
print("Statistical Summary")
print(df.describe())

print("\n")
print("=" * 50)
print("Class Distribution")
print(df["Class"].value_counts())

# -----------------------------
# Plot Class Distribution
# -----------------------------
plt.figure(figsize=(6,4))

df["Class"].value_counts().plot(kind="bar")

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.savefig("images/class_distribution.png")

plt.show()