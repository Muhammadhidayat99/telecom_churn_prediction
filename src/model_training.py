from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train):
    """Train and return logistic regression model"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train and return random forest model"""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train and return XGBoost model"""
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_xgboost(X_train, y_train):
    """Hyperparameter tuning for XGBoost"""
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_