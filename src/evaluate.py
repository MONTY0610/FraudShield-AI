import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# -----------------------------
# Create folders if not present
# -----------------------------
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/creditcard.csv/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------
# Scale Features
# -----------------------------
scaler = StandardScaler()

X["Time"] = scaler.fit_transform(X[["Time"]])
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Load Saved Model
# -----------------------------
model = joblib.load("models/fraud_model.pkl")

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

metrics.to_csv("reports/model_metrics.csv", index=False)

print(metrics)

# -----------------------------
# Confusion Matrix
# -----------------------------
plt.figure(figsize=(6,6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# ROC Curve
# -----------------------------
plt.figure(figsize=(6,6))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve")

plt.savefig(
    "images/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Precision Recall Curve
# -----------------------------
plt.figure(figsize=(6,6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("Precision Recall Curve")

plt.savefig(
    "images/precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Feature Importance
# -----------------------------
if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Important Features")

    plt.savefig(
        "images/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nEvaluation Completed Successfully!")