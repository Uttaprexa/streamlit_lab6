"""
Generate sample heart disease dataset
"""
import pandas as pd
import numpy as np
import os

def main():
    print("Generating heart disease dataset...")
    
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'age': np.random.randint(30, 80, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(0, 4, n),
        'trestbps': np.random.randint(90, 200, n),
        'chol': np.random.randint(120, 400, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.randint(70, 200, n),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.random.uniform(0, 6, n),
        'slope': np.random.randint(0, 3, n),
        'ca': np.random.randint(0, 4, n),
        'thal': np.random.randint(0, 4, n),
    })
    
    df['target'] = (
        ((df['age'] > 55) & (df['chol'] > 250)) | 
        (df['cp'] > 2) | 
        (df['thalach'] < 120)
    ).astype(int)
    
    df.to_csv('data/heart_disease.csv', index=False)
    print(f"✓ Dataset created: {df.shape}")

if __name__ == "__main__":
    main()