import streamlit as st
import plotly.express as px
from src.utils import load_dataset, load_feature_importance

st.set_page_config(
    page_title="Analytics",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# Header
# ==========================================

st.title("📈 Dataset Analytics")

st.markdown("""
Explore the characteristics of the credit card transaction dataset through
interactive visualizations and gain insights into fraud patterns, feature
importance, and transaction distributions.
""")

st.divider()

# ==========================================
# Load Data
# ==========================================

df = load_dataset()
feature_df = load_feature_importance()

total = len(df)
frauds = len(df[df["Class"] == 1])
normal = len(df[df["Class"] == 0])
fraud_rate = (frauds / total) * 100

# ==========================================
# Executive Summary
# ==========================================

st.subheader("📊 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions", f"{total:,}")
col2.metric("Frauds", frauds)
col3.metric("Normal", f"{normal:,}")
col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

st.divider()

# ==========================================
# Row 1
# ==========================================

left, right = st.columns(2)

with left:

    st.subheader("📊 Class Distribution")

    class_df = (
        df["Class"]
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )

    class_df["Class"] = class_df["Class"].map({
        0: "Normal",
        1: "Fraud"
    })

    fig = px.bar(
        class_df,
        x="Class",
        y="Count",
        text="Count",
        color="Class",
        title="Fraud vs Normal Transactions",
        color_discrete_map={
            "Normal": "#22C55E",
            "Fraud": "#EF4444"
        }
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader("💰 Transaction Amount Distribution")

    fig = px.histogram(
        df,
        x="Amount",
        nbins=80,
        title="Transaction Amount Distribution",
        color_discrete_sequence=["#4F46E5"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================
# Row 2
# ==========================================

left, right = st.columns(2)

with left:

    st.subheader("⭐ Feature Importance")

    top = (
        feature_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        top,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top 15 Important Features"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader("🔥 Correlation Heatmap")

    corr = (
        df
        .drop(columns=["Class"])
        .corr(numeric_only=True)
    )

    fig = px.imshow(
        corr,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================
# Business Insights
# ==========================================

st.subheader("💡 Business Insights")

st.info(f"""
### Key Findings

• **Total Transactions:** {total:,}

• **Fraud Transactions:** {frauds}

• **Fraud Rate:** {fraud_rate:.3f}%

• Fraud transactions account for less than **1%** of the dataset, making this a highly imbalanced classification problem.

• XGBoost was selected as the production model after outperforming Logistic Regression and Random Forest.

• The application generates fraud probability, risk score, and business recommendations for every transaction.
""")

st.divider()

st.caption("Developed by Tanay | FraudShield AI v1.0 | 2026")

st.divider()

st.caption(
    "© 2026 FraudShield AI | Built using Python, XGBoost and Streamlit"
)