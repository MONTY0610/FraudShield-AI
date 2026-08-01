import streamlit as st

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("💳 FraudShield AI")

st.subheader("Intelligent Credit Card Fraud Detection System")

st.markdown("---")

st.markdown("""
## Welcome 👋

FraudShield AI is a Machine Learning application that detects fraudulent credit card transactions using XGBoost.

### Features

- 📊 Interactive Dashboard
- 📈 Data Analytics
- 🤖 Single Transaction Prediction
- 📂 Batch Prediction
- 📉 Model Performance
- 📑 Download Predictions
- 📋 Professional Reports

Use the sidebar to navigate through the application.
""")

st.success("Model Loaded Successfully ✅")
