import joblib
import pandas as pd

from src.utils import load_model

# ==========================================
# Load Model and Scalers
# ==========================================

model = load_model()

time_scaler = joblib.load("models/time_scaler.pkl")
amount_scaler = joblib.load("models/amount_scaler.pkl")


# ==========================================
# Preprocess Input
# ==========================================

def preprocess_input(df):
    """
    Scale the Time and Amount columns
    before making predictions.
    """

    df = df.copy()

    if "Time" in df.columns:
        df["Time"] = time_scaler.transform(df[["Time"]])

    if "Amount" in df.columns:
        df["Amount"] = amount_scaler.transform(df[["Amount"]])

    return df


# ==========================================
# Prediction Function
# ==========================================

def predict(df):
    """
    Predict fraudulent transactions and
    generate business-friendly risk analysis.
    """

    # --------------------------------------
    # Validate Input
    # --------------------------------------

    required_columns = (
        ["Time", "Amount"] +
        [f"V{i}" for i in range(1, 29)]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------
    # Preprocess Data
    # --------------------------------------

    processed_df = preprocess_input(df)

    # --------------------------------------
    # Model Prediction
    # --------------------------------------

    predictions = model.predict(processed_df)

    probabilities = model.predict_proba(processed_df)[:, 1]

    # --------------------------------------
    # Prepare Results
    # --------------------------------------

    result = df.copy()

    result["Prediction"] = predictions

    result["Fraud Probability"] = (
        probabilities * 100
    ).round(2)

    risk_scores = (
        probabilities * 100
    ).round().astype(int)

    result["Risk Score"] = risk_scores

    # --------------------------------------
    # Risk Level & Recommendation
    # --------------------------------------

    risk_levels = []
    recommendations = []

    for score in risk_scores:

        if score >= 80:
            risk_levels.append("🔴 HIGH")
            recommendations.append(
                "Decline Transaction"
            )

        elif score >= 40:
            risk_levels.append("🟡 MEDIUM")
            recommendations.append(
                "Verify using OTP"
            )

        else:
            risk_levels.append("🟢 LOW")
            recommendations.append(
                "Approve Transaction"
            )

    result["Risk Level"] = risk_levels
    result["Recommendation"] = recommendations

    return result