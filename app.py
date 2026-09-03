import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/fraud_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('data/credit_card_fraud.csv')

def simulate_pca_transformation(df):
    """
    Simulates the PCA transformation.
    In a real app, this would use a saved PCA transformer.
    Here, we ensure the data matches the model's expected input (30 features).
    """
    processed_df = df.copy()

    # Ensure Time and Amount exist
    if 'Time' not in processed_df.columns:
        processed_df['Time'] = 0
    if 'Amount' not in processed_df.columns:
        processed_df['Amount'] = 0

    # Simulate V1-V28:
    # If they don't exist, we generate them based on the provided data
    for i in range(1, 29):
        col_name = f'V{i}'
        if col_name not in processed_df.columns:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                processed_df[col_name] = processed_df[numeric_cols].mean(axis=1) * 0.1 + np.random.randn(len(processed_df))
            else:
                processed_df[col_name] = np.random.randn(len(processed_df))

    # Return only the 30 features in the correct order
    cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    return processed_df[cols]

try:
    model, scaler = load_artifacts()
    try:
        df_full = load_data()
        data_available = True
    except Exception:
        df_full = None
        data_available = False
except Exception as e:
    st.error(f"Critical Error: Could not load model artifacts: {e}. Please run src/train_pipeline.py first.")
    st.stop()

# Title and Description
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("""
Welcome to the Professional Fraud Detection Suite. You can perform **single transaction checks** or **bulk analyze** entire datasets.
""")

# Create Tabs for different modes
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Bulk Analysis"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")

        with st.container(border=True):
            st.markdown("### Primary Details")
            time = st.number_input("Transaction Time (seconds)", min_value=0, max_value=172792, value=0)
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=10.0)

            with st.expander("⚙️ Advanced Settings (PCA Features)"):
                st.markdown("""
                *These features (V1-V28) are anonymized system components generated via Principal Component Analysis (PCA).
                In a real app, these are calculated automatically from raw data.*
                """)
                v_features = {}
                for i in range(1, 29):
                    v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.4f")

        if st.button("Predict Transaction", type="primary"):
            input_df = pd.DataFrame([[time, amount] + list(v_features.values())],
                                    columns=['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)])

            scaled_input = input_df.copy()
            scaled_input[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            if prediction == 1:
                st.error(f"🚨 **Fraudulent Transaction Detected!**")
                st.write(f"Probability of Fraud: {probability:.2%}")
            else:
                st.success(f"✅ **Legitimate Transaction**")
                st.write(f"Probability of Fraud: {probability:.2%}")

    with col2:
        st.subheader("Analysis")
        if data_available:
            st.markdown("**Amount Distribution**")
            fig, ax = plt.subplots()
            sns.histplot(df_full['Amount'], bins=50, kde=True, ax=ax, color='blue')
            ax.axvline(amount, color='red', linestyle='--', label='User Input')
            ax.set_title("Transaction Amount Distribution")
            ax.legend()
            st.pyplot(fig)

            st.markdown("**Feature Analysis (V1-V5)**")
            v_cols = ['V1', 'V2', 'V3', 'V4', 'V5']
            mean_legit = df_full[df_full['Class'] == 0][v_cols].mean()
            mean_fraud = df_full[df_full['Class'] == 1][v_cols].mean()
            user_vals = pd.Series({k: v_features.get(k, 0.0) for k in v_cols})

            comp_df = pd.DataFrame({
                'Legitimate (Avg)': mean_legit,
                'Fraudulent (Avg)': mean_fraud,
                'User Input': user_vals
            })
            st.bar_chart(comp_df)
        else:
            st.warning("Dataset not found. Visualizations are disabled, but predictions will still work.")

with tab2:
    st.subheader("Bulk Transaction Analysis")
    st.markdown("Upload a CSV file containing your transactions. The system will automatically handle the PCA transformation and predict fraud for every row.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview", raw_df.head())

        if st.button("Run Bulk Analysis", type="primary"):
            with st.spinner("Transforming data and calculating predictions..."):
                transformed_df = simulate_pca_transformation(raw_df)
                scaled_df = transformed_df.copy()
                scaled_df[['Time', 'Amount']] = scaler.transform(transformed_df[['Time', 'Amount']])

                predictions = model.predict(scaled_df)
                probabilities = model.predict_proba(scaled_df)[:, 1]

                result_df = raw_df.copy()
                result_df['Fraud_Prediction'] = predictions
                result_df['Fraud_Probability'] = probabilities

                fraud_count = sum(predictions)
                total_count = len(predictions)

                st.divider()
                st.subheader("Analysis Results")

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Total Transactions", total_count)
                m_col2.metric("Fraudulent Detected", fraud_count, delta=f"{(fraud_count/total_count):.2%}", delta_color="inverse")
                m_col3.metric("Legitimate", total_count - fraud_count)

                st.write("### Detailed Results (First 100 rows)")
                st.dataframe(result_df.head(100))

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption("Developed for Fraud Detection Project Demo")
