# 🛡️ Credit Card Fraud Detection Project

This project implements a complete machine learning pipeline to detect fraudulent credit card transactions using a synthetic dataset. It includes data generation, a training pipeline with SMOTE for handling class imbalance, and an interactive Streamlit dashboard for real-time predictions.

## 🚀 Project Structure

```text
fraud_detection_project/
├── data/                   # Synthetic datasets
├── models/                 # Persisted model and scaler
├── src/                    # Pipeline source code
│   ├── data_generation.py  # Dataset creation script
│   └── train_pipeline.py    # Training and evaluation script
├── app.py                  # Streamlit dashboard
├── requirements.txt       # Python dependencies
└── README.md               # Instructions
```

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone <https://github.com/Abhinandan305/Fraud-Detection>
cd "Fraud Detection"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data and Train Model
Run the scripts in the following order:
```bash
# Generate synthetic dataset
python src/data_generation.py

# Train the model and save artifacts
python src/train_pipeline.py
```
This will create the `data/` and `models/` folders.

### 4. Run the Dashboard
```bash
streamlit run app.py
```

## 📊 Model Details
- **Algorithm**: Random Forest Classifier.
- **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique).
- **Scaling**: Standard Scaling for `Time` and `Amount`.
- **Evaluation**: Focused on Precision-Recall curves and AUPRC due to extreme class imbalance.
