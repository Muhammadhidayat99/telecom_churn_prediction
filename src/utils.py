import pickle
import json
import os
from datetime import datetime

def save_model(model, model_name, directory='models'):
    """Save trained model to file"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/{model_name}_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved as {filename}")
    return filename

def load_model(filepath):
    """Load saved model from file"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def save_metrics(metrics, model_name, directory='reports'):
    """Save evaluation metrics to JSON file"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/{model_name}_metrics_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved as {filename}")
    return filename

def log_experiment(experiment_details, directory='reports'):
    """Log experiment details"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/experiment_log_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(experiment_details, f, indent=4)
    
    print(f"Experiment logged as {filename}")
    return filename