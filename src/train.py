import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data/creditcard.csv/creditcard.csv")

# -------------------------
# Features & Target
# -------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------
# Scale Time and Amount
# -------------------------
time_scaler = StandardScaler()
amount_scaler = StandardScaler()

X["Time"] = time_scaler.fit_transform(X[["Time"]])
X["Amount"] = amount_scaler.fit_transform(X[["Amount"]])

joblib.dump(time_scaler, "models/time_scaler.pkl")
joblib.dump(amount_scaler, "models/amount_scaler.pkl")

# -------------------------
# Train Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )
}

best_model = None
best_auc = 0

print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model

# -------------------------
# Save Best Model
# -------------------------
joblib.dump(best_model, "models/fraud_model.pkl")

print("\n")
print("=" * 70)
print("Best model saved successfully!")