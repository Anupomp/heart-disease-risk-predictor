import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns for consistency
    df.rename(columns={
        'cp': 'chest_pain',
        'trestbps': 'resting_bp',
        'chol': 'cholesterol',
        'fbs': 'fasting_bs',
        'restecg': 'resting_ecg',
        'thalch': 'max_hr',
        'exang': 'exercise_angina',
        'oldpeak': 'oldpeak',
        'slope': 'st_slope',
        'ca': 'ca',
        'thal': 'thal',
        'num': 'target'  # Binary target: 0 (no disease), 1+ (disease)
    }, inplace=True)

    # Convert target to binary: 0 (no disease), 1 (disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Drop irrelevant columns
    df.drop(columns=['dataset'], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Convert categorical columns
    categorical_cols = ['sex', 'chest_pain', 'resting_ecg', 'exercise_angina', 'st_slope', 'thal']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # Encode categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Cast binary column
    df['fasting_bs'] = df['fasting_bs'].astype(int)

    # Handle outliers with Z-score method
    num_cols = ['age', 'resting_bp', 'cholesterol', 'max_hr', 'oldpeak']
    for col in num_cols:
        z = (df[col] - df[col].mean()) / df[col].std()
        df = df[z.abs() <= 3]

    # Normalize numerical features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
