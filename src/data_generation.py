import pandas as pd
import numpy as np
import os

def generate_fraud_data(n_samples=10000, fraud_ratio=0.0017):
    """
    Generates a synthetic credit card fraud dataset.

    Args:
        n_samples (int): Total number of transactions.
        fraud_ratio (float): Proportion of transactions that are fraudulent.

    Returns:
        pd.DataFrame: Synthetic fraud dataset.
    """
    print(f"Generating {n_samples} samples with a fraud ratio of {fraud_ratio}...")

    # Number of fraud and legitimate samples
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # 1. Generate Legitimate Transactions
    # Time: Random between 0 and 172,792 (approx 2 days)
    time_legit = np.random.randint(0, 172792, n_legit)
    # Amount: Log-normal distribution to mimic real spending
    amount_legit = np.random.lognormal(mean=3, sigma=1, size=n_legit)
    # V1-V28: Random normal distribution (mean=0, std=1)
    v_legit = np.random.randn(n_legit, 28)

    # 2. Generate Fraudulent Transactions
    time_fraud = np.random.randint(0, 172792, n_fraud)
    # Fraud often has different amount distributions (e.g., very small or very large)
    amount_fraud = np.random.lognormal(mean=4, sigma=1.5, size=n_fraud)
    # V1-V28: Shifted normal distribution to make fraud detectable
    # We shift some features to create patterns
    v_fraud = np.random.randn(n_fraud, 28)
    v_fraud[:, 0] -= 2  # Shift V1
    v_fraud[:, 5] += 3  # Shift V6
    v_fraud[:, 12] -= 1 # Shift V13

    # Combine the data
    data_legit = np.column_stack([time_legit, amount_legit, v_legit])
    data_fraud = np.column_stack([time_fraud, amount_fraud, v_fraud])

    X = np.vstack([data_legit, data_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    # Create column names
    cols = ['Time', 'Amount'] + [f'V{i+1}' for i in range(28)]
    df = pd.DataFrame(X, columns=cols)
    df['Class'] = y

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Generate data
    df_fraud = generate_fraud_data(n_samples=100000, fraud_ratio=0.0017)

    # Save to CSV
    output_path = 'data/credit_card_fraud.csv'
    df_fraud.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to {output_path}")
    print(f"Class distribution:\n{df_fraud['Class'].value_counts(normalize=True)}")
