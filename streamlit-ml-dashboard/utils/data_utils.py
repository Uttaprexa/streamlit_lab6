"""
Data utility functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_heart_disease_data(filepath='data/heart_disease.csv'):
    """Load heart disease dataset"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for training"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_feature_names():
    """Get feature names and descriptions"""
    features = {
        'age': 'Age in years',
        'sex': 'Sex (0=Female, 1=Male)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (0=No, 1=Yes)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (0=No, 1=Yes)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels (0-3)',
        'thal': 'Thalassemia (0-3)'
    }
    return features

def calculate_statistics(df):
    """Calculate dataset statistics"""
    stats = {
        'total_samples': len(df),
        'n_features': df.shape[1] - 1,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'disease_cases': (df['target'] == 1).sum(),
        'healthy_cases': (df['target'] == 0).sum(),
        'disease_rate': (df['target'] == 1).mean() * 100
    }
    return stats

def get_correlation_with_target(df):
    """Get feature correlations with target"""
    corr = df.corr()['target'].drop('target').sort_values(ascending=False)
    return corr