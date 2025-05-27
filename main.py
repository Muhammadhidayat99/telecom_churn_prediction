from src.data_preprocessing import load_data, preprocess_data, split_and_scale_data
from src.feature_engineering import handle_class_imbalance
from src.model_training import train_logistic_regression, train_random_forest, train_xgboost, tune_xgboost
from src.model_evaluation import evaluate_model, plot_feature_importance
from src.utils import save_model, save_metrics

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data('D:\internship\telecom_churn_prediction\data\telecom_churn_mock_data.csv')
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df)
    
    # Handle class imbalance
    print("\nHandling class imbalance...")
    X_train_res, y_train_res = handle_class_imbalance(X_train, y_train)
    
    # Train models
    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train_res, y_train_res)
    evaluate_model(lr_model, X_test, y_test)
    save_model(lr_model, 'logistic_regression')
    
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train_res, y_train_res)
    evaluate_model(rf_model, X_test, y_test)
    plot_feature_importance(rf_model, X_train.columns)
    save_model(rf_model, 'random_forest')
    
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train_res, y_train_res)
    evaluate_model(xgb_model, X_test, y_test)
    plot_feature_importance(xgb_model, X_train.columns)
    save_model(xgb_model, 'xgboost')
    
    print("\nTuning XGBoost...")
    tuned_xgb = tune_xgboost(X_train_res, y_train_res)
    evaluate_model(tuned_xgb, X_test, y_test)
    plot_feature_importance(tuned_xgb, X_train.columns)
    save_model(tuned_xgb, 'xgboost_tuned')

if __name__ == "__main__":
    main()