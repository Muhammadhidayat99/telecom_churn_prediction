from imblearn.over_sampling import SMOTE

def handle_class_imbalance(X_train, y_train, random_state=42):
    """Apply SMOTE to handle class imbalance"""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def create_interaction_features(df):
    """Create meaningful interaction features"""
    # Example: Create tenure segments
    df['TenureSegment'] = pd.cut(df['Tenure'], 
                                bins=[0, 12, 24, 36, 48, 60, 72],
                                labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5-6'])
    
    # Example: Monthly charge per tenure
    df['MonthlyChargePerTenure'] = df['MonthlyCharges'] / (df['Tenure'] + 1)  # +1 to avoid division by zero
    
    return df

def select_features(df, method='correlation', threshold=0.1):
    """Feature selection methods"""
    if method == 'correlation':
        corr_matrix = df.corr()
        churn_corr = corr_matrix['Churn'].abs().sort_values(ascending=False)
        selected_features = churn_corr[churn_corr > threshold].index.tolist()
        return df[selected_features]
    
    # Could add other feature selection methods
    return df