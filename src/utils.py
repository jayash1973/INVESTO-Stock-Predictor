import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    
    Args:
        file_path (str): Path to save the object
        obj: Python object to be saved
        
    Raises:
        CustomException: If any error occurs during saving
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load a Python object from a file using pickle.
    
    Args:
        file_path (str): Path to the file to load
        
    Returns:
        The loaded Python object
        
    Raises:
        CustomException: If any error occurs during loading
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_data(ticker, artifacts_dir="artifacts"):
    """
    Load all datasets and scaler for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        artifacts_dir (str): Directory where artifacts are stored
        
    Returns:
        dict: Dictionary containing all datasets and scaler
        
    Raises:
        CustomException: If any error occurs during loading
    """
    try:
        datasets = {}
        
        # Helper function to load CSV with consistent settings
        def load_csv(filename):
            return pd.read_csv(
                os.path.join(artifacts_dir, f"{ticker}_{filename}.csv"),
                index_col=0,
                parse_dates=True
            )
        
        # Load train and test sets
        datasets['X_train'] = load_csv("X_train")
        datasets['X_test'] = load_csv("X_test")
        datasets['y_train'] = load_csv("y_train").squeeze("columns")
        datasets['y_test'] = load_csv("y_test").squeeze("columns")
        datasets['X_train_scaled'] = load_csv("X_train_scaled")
        datasets['X_test_scaled'] = load_csv("X_test_scaled")
        
        # Load full engineered data for time series analysis
        datasets['full_data'] = load_csv("engineered")
        
        # Load scaler
        scaler_path = os.path.join(artifacts_dir, f"{ticker}_scaler.pkl")
        datasets['scaler'] = load_object(scaler_path)
        
        print(f"Successfully loaded data for {ticker}")
        return datasets
    except Exception as e:
        raise CustomException(e, sys)

