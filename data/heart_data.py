import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_heart_data():
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=13,
        n_informative=10,
        n_redundant=3,
        random_state=42
    )
    feature_names = [
        'age', 'sex', 'chest_pain', 'resting_bp',
        'cholesterol', 'fasting_bs', 'rest_ecg',
        'max_hr', 'exercise_angina', 'st_depression',
        'st_slope', 'num_vessels', 'thal'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

def split_data_for_clients(df, num_clients=3):
    client_data = []
    X_all = df.drop('target', axis=1).values
    y_all = df['target'].values
    indices = np.array_split(np.arange(len(X_all)), num_clients)

    for i, idx in enumerate(indices):
        X = X_all[idx]
        y = y_all[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        client_data.append({
            'client_id': i + 1,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        })
        print(f"Client {i+1}: {len(X_train)} train samples, {len(X_test)} test samples")
    return client_data

if __name__ == "__main__":
    df = load_heart_data()
    print(f"Dataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    client_data = split_data_for_clients(df)
    print("\nData successfully split for 3 clients!")