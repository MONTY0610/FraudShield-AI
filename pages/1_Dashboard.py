import streamlit as st
from src.utils import load_dataset, load_metrics

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Load Data
# ==========================================

df = load_dataset()
metrics = load_metrics()

total = len(df)
frauds = len(df[df["Class"] == 1])
normal = len(df[df["Class"] == 0])
fraud_rate = (frauds / total) * 100

training = int(total * 0.8)
testing = total - training

accuracy = metrics.loc[
    metrics["Metric"] == "Accuracy",
    "Value"
].values[0]

precision = metrics.loc[
    metrics["Metric"] == "Precision",
    "Value"
].values[0]

recall = metrics.loc[
    metrics["Metric"] == "Recall",
    "Value"
].values[0]

roc_auc = metrics.loc[
    metrics["Metric"] == "ROC AUC",
    "Value"
].values[0]

# ==========================================
# Header
# ==========================================

st.title("💳 FraudShield AI")

st.markdown("""
### Intelligent Credit Card Fraud Detection System

An end-to-end Machine Learning application that detects fraudulent
credit card transactions using **XGBoost**, provides **risk scoring**,
**business recommendations**, and **interactive analytics**.
""")

st.divider()

# ==========================================
# System Status
# ==========================================

st.subheader("🟢 System Status")

left, right = st.columns(2)

with left:
    st.success("""
✅ Model Loaded

✅ Prediction Engine Active

✅ Risk Engine Active

✅ Dataset Connected
""")

with right:
    st.info("""
**Production Model:** XGBoost

**Version:** 1.0

**Status:** Operational

**Platform:** Streamlit
""")

st.divider()

# ==========================================
# Model Performance
# ==========================================

st.subheader("📈 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
c2.metric("ROC AUC", f"{roc_auc * 100:.2f}%")
c3.metric("Precision", f"{precision * 100:.2f}%")
c4.metric("Recall", f"{recall * 100:.2f}%")

st.divider()

# ==========================================
# Dataset Overview
# ==========================================

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions", f"{total:,}")
c2.metric("Normal", f"{normal:,}")
c3.metric("Frauds", frauds)
c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

st.divider()

# ==========================================
# Model Information
# ==========================================

st.subheader("🤖 Model Information")

left, right = st.columns(2)

with left:
    st.write("**Algorithm:** XGBoost")
    st.write("**Features:** 30")
    st.write(f"**Training Samples:** {training:,}")
    st.write(f"**Testing Samples:** {testing:,}")

with right:
    st.write("**Dataset:** Credit Card Fraud Detection")
    st.write("**Prediction Type:** Binary Classification")
    st.write("**Output:** Fraud Probability")
    st.write("**Risk Engine:** Enabled")

st.divider()

# ==========================================
# Executive Summary
# ==========================================

st.subheader("📋 Executive Summary")

st.success(f"""
### Current System Summary

- **Total Transactions:** {total:,}
- **Fraud Transactions:** {frauds}
- **Fraud Rate:** {fraud_rate:.3f}%
- **Model Accuracy:** {accuracy * 100:.2f}%
- **ROC AUC Score:** {roc_auc * 100:.2f}%

The trained XGBoost model demonstrates excellent performance and is ready for fraud prediction tasks.
""")

# ==========================================
# Quick Insights
# ==========================================

st.subheader("💡 Quick Insights")

st.info(f"""
- Fraud transactions account for only **{fraud_rate:.3f}%** of the dataset.

- The dataset is highly imbalanced, making fraud detection a challenging machine learning problem.

- XGBoost was selected after comparing Logistic Regression and Random Forest.

- The prediction engine provides fraud probability, risk score, and business recommendations for every transaction.
""")

st.divider()

# ==========================================
# Sample Dataset
# ==========================================

st.subheader("📄 Sample Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.divider()

st.caption("Developed by Tanay | FraudShield AI v1.0 | 2026")

st.divider()

st.caption(
    "© 2026 FraudShield AI | Built using Python, XGBoost and Streamlit"
)