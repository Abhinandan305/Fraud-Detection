import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

def run_pipeline():
    # 1. Load Data
    data_path = 'data/credit_card_fraud.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print("Data loaded successfully.")

    # 2. Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Stratified split to maintain fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale Time and Amount (V features are already synthetic normal)
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    # 3. Handle Class Imbalance using SMOTE
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training set size: {X_resampled.shape}")

    # 4. Model Training
    print("Training Random Forest Classifier...")
    # Using class_weight='balanced' as an extra layer of protection
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_resampled, y_resampled)

    # 5. Evaluation
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"Area Under Precision-Recall Curve (AUPRC): {pr_auc:.4f}")

    # Plotting results
    plt.figure(figsize=(12, 5))

    # PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig('models/evaluation_metrics.png')
    print("Evaluation plots saved to models/evaluation_metrics.png")

    # 6. Persistence
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/fraud_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Model and Scaler saved to models/ directory.")

if __name__ == "__main__":
    run_pipeline()
