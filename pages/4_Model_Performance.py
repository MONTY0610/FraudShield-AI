import streamlit as st
from PIL import Image
from src.utils import load_metrics

st.set_page_config(
    page_title="Model Insights",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# Header
# =====================================================

st.title("📈 Model Insights")

st.markdown("""
This page presents the evaluation of the final **XGBoost** model using
multiple performance metrics and visualizations generated during model evaluation.
""")

st.divider()

# =====================================================
# Model Metrics
# =====================================================

st.subheader("📊 Model Performance Metrics")

metrics = load_metrics()

col1, col2, col3, col4 = st.columns(4)

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

col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Precision", f"{precision*100:.2f}%")
col3.metric("Recall", f"{recall*100:.2f}%")
col4.metric("ROC AUC", f"{roc_auc*100:.2f}%")

st.dataframe(
    metrics,
    use_container_width=True
)

st.divider()

# =====================================================
# Evaluation Plots
# =====================================================

left, right = st.columns(2)

with left:

    st.subheader("🧩 Confusion Matrix")

    confusion = Image.open("images/confusion_matrix.png")

    st.image(
        confusion,
        use_container_width=True
    )

with right:

    st.subheader("📈 ROC Curve")

    roc = Image.open("images/roc_curve.png")

    st.image(
        roc,
        use_container_width=True
    )

st.divider()

left, right = st.columns(2)

with left:

    st.subheader("📉 Precision-Recall Curve")

    pr = Image.open(
        "images/precision_recall_curve.png"
    )

    st.image(
        pr,
        use_container_width=True
    )

with right:

    st.subheader("⭐ Feature Importance")

    feature = Image.open(
        "images/feature_importance.png"
    )

    st.image(
        feature,
        use_container_width=True
    )

st.divider()

# =====================================================
# Business Interpretation
# =====================================================

st.subheader("💡 Business Interpretation")

st.success("""
### Key Observations

- The XGBoost model achieved excellent predictive performance with very high **Accuracy**, **Precision**, and **ROC-AUC**.

- High Precision indicates that very few legitimate transactions are incorrectly classified as fraudulent.

- Strong Recall enables the model to identify most fraudulent transactions successfully.

- XGBoost outperformed Logistic Regression and Random Forest during model comparison and was selected as the final production model.
""")

st.info("""
### Business Impact

- 💳 Faster fraud detection

- 💰 Reduced financial losses

- 🎯 Lower false positives

- 😊 Better customer experience

- ⚡ Real-time risk assessment support

This application is intended to assist fraud analysts by providing decision-support rather than replacing manual review processes.
""")

st.divider()

st.caption(
    "Developed by Tanay | FraudShield AI v1.0 | 2026"
)

st.divider()

st.caption(
    "© 2026 FraudShield AI | Built using Python, XGBoost and Streamlit"
)