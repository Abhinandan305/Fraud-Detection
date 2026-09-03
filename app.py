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

try:
    model, scaler = load_artifacts()
    # Try to load data, but don't crash if it's missing
    try:
        df = load_data()
        data_available = True
    except Exception:
        df = None
        data_available = False
except Exception as e:
    st.error(f"Critical Error: Could not load model artifacts: {e}. Please run src/train_pipeline.py first.")
    st.stop()

# Title and Description
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("""
This application uses a Machine Learning model (Random Forest) to detect potentially fraudulent transactions.
Input the transaction details on the left and see the prediction on the right.
""")

# Sidebar for inputs
st.sidebar.header("Transaction Details")

def user_input_features():
    # Time and Amount
    st.sidebar.subheader("Primary Details")
    time = st.sidebar.number_input("Transaction Time (seconds)", min_value=0, max_value=172792, value=0)
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=10.0)

    # V1-V28 in an expander to reduce clutter
    with st.sidebar.expander("⚙️ Advanced Settings (PCA Features)"):
        st.markdown("""
        *These features (V1-V28) are anonymized system components generated via Principal Component Analysis (PCA).
        In a real app, these are calculated automatically from raw data.*
        """)

        v_features = {}
        for i in range(1, 29):
            v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.4f")

    # Combine into a DataFrame
    input_df = pd.DataFrame([[time, amount] + list(v_features.values())],
                            columns=['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)])
    return input_df

input_df = user_input_features()

# Main Panel
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Summary")
    st.write(input_df)

    # Prediction
    if st.button("Predict Transaction"):
        # Preprocess input
        # We only scale Time and Amount as per the pipeline
        scaled_input = input_df.copy()
        scaled_input[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

        # Predict
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
        # Visualization: Amount Distribution
        st.markdown("**Amount Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(df['Amount'], bins=50, kde=True, ax=ax, color='blue')
        ax.axvline(input_df['Amount'][0], color='red', linestyle='--', label='User Input')
        ax.set_title("Transaction Amount Distribution")
        ax.legend()
        st.pyplot(fig)

        # Visualization: Feature Comparison (simplified)
        st.markdown("**Feature Analysis (V1-V5)**")
        # Compare user input to mean of legit and fraud
        v_cols = ['V1', 'V2', 'V3', 'V4', 'V5']
        mean_legit = df[df['Class'] == 0][v_cols].mean()
        mean_fraud = df[df['Class'] == 1][v_cols].mean()
        user_vals = input_df[v_cols].iloc[0]

        comp_df = pd.DataFrame({
            'Legitimate (Avg)': mean_legit,
            'Fraudulent (Avg)': mean_fraud,
            'User Input': user_vals
        })

        st.bar_chart(comp_df)
    else:
        st.warning("Dataset not found. Visualizations are disabled, but predictions will still work.")

# Footer
st.markdown("---")
st.caption("Developed for Fraud Detection Project Demo")
