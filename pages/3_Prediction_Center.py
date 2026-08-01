import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict import predict

st.set_page_config(
    page_title="Prediction Center",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# Header
# ==================================================

st.title("🤖 Prediction Center")

st.markdown("""
Upload a CSV file containing credit card transactions to generate
fraud predictions, risk scores, and business recommendations.
""")

st.divider()

# ==================================================
# Upload File
# ==================================================

uploaded_file = st.file_uploader(
    "📂 Upload a CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.write(f"**Total Transactions:** {len(df):,}")

    if st.button("🚀 Predict Fraud"):

        with st.spinner("Running prediction..."):

            input_df = df.copy()

            if "Class" in input_df.columns:
                input_df = input_df.drop(columns=["Class"])

            try:
                results = predict(input_df)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.success("✅ Prediction completed successfully. Risk analysis report generated.")

        # ==================================================
        # Summary Metrics
        # ==================================================

        frauds = len(results[results["Prediction"] == 1])
        safe = len(results[results["Prediction"] == 0])
        high = len(results[results["Risk Level"] == "🔴 HIGH"])

        average_probability = results["Fraud Probability"].mean()

        st.subheader("📊 Prediction Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🚨 Frauds",
                frauds
            )

        with col2:
            st.metric(
                "✅ Safe",
                safe
            )

        with col3:
            st.metric(
                "🔴 High Risk",
                high
            )

        with col4:
            st.metric(
                "📈 Avg Probability",
                f"{average_probability:.2f}%"
            )

        st.divider()

        # ==================================================
        # Risk Distribution
        # ==================================================

        st.subheader("🥧 Risk Distribution")

        summary = (
            results["Risk Level"]
            .value_counts()
            .reset_index()
        )

        summary.columns = [
            "Risk Level",
            "Count"
        ]

        fig = px.pie(
            summary,
            names="Risk Level",
            values="Count",
            hole=0.45,
            title="Prediction Distribution",
            color="Risk Level",
            color_discrete_map={
                "🟢 LOW": "#22C55E",
                "🟡 MEDIUM": "#FACC15",
                "🔴 HIGH": "#EF4444"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # Business Summary
        # ==================================================

        st.subheader("📋 Business Summary")

        st.info(f"""
### Analysis Report

- **Total Transactions:** {len(results):,}

- **Fraud Transactions:** {frauds}

- **Safe Transactions:** {safe}

- **High Risk Transactions:** {high}

### Recommendation

- 🔴 HIGH → Decline transaction or perform manual review.

- 🟡 MEDIUM → Verify customer using OTP or additional authentication.

- 🟢 LOW → Transaction can be safely approved.
""")

        st.warning("""
⚠ This application is intended to assist fraud analysts.

Predictions should support decision-making and not replace existing fraud prevention systems.
""")

        st.divider()

        # ==================================================
        # Prediction Results
        # ==================================================

        st.subheader("📄 Prediction Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        st.download_button(
            label="📥 Export Analysis Report",
            data=results.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

        st.divider()

        st.caption(
            "Developed by Tanay | FraudShield AI v1.0 | 2026"
        )

        st.divider()

st.caption(
    "© 2026 FraudShield AI | Built using Python, XGBoost and Streamlit"
)