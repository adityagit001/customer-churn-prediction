import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_and_preprocess(df):
    df = df.copy()
    
    # 1. Drop customerID (unique identifier)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    # 2. TotalCharges column has whitespaces representing missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # 3. Target Encoding: Churn (Yes -> 1, No -> 0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 4. Binary columns conversion
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # 5. One-Hot Encoding for categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 6. Split Features and Target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 7. Train-Test Split (stratified due to churn class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 8. Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()