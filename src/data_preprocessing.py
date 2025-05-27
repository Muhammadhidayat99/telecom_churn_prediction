import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load and initial processing of raw data"""
    df = pd.read_csv('D:/internship/telecom_churn_prediction/data/telecom_churn_mock_data.csv')
    
    # Handle missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['Tenure'])
    df = df.dropna()
    
    return df

def preprocess_data(df):
    """Feature engineering and preprocessing"""
    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'Yes': 1, 'No': 0}))
    
    # One-hot encoding
    cat_cols_to_encode = ['Gender','InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True)
    
    # Other categoricals
    other_cat_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in other_cat_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop customer ID
    df = df.drop('CustomerID', axis=1)
    
    return df

def split_and_scale_data(df, test_size=0.2, random_state=42):
    """Split data and scale numerical features"""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Scale numerical features
    num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, scaler