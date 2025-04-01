import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import time
import pickle
from datetime import datetime, timedelta

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from itertools import product
from dataclasses import dataclass 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Set up plotting styles
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
warnings.filterwarnings("ignore")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    time_series_model_path: str = os.path.join("artifacts", "time_series_model.pkl")

class ModelTrainer:
    def __init__(self):
        super().__init__() 
        self.model_trainer_config = ModelTrainerConfig()
        self.tickers = ['AAPL', 'GOOGL', 'META', 'MSFT', 'NVDA']
        logging.info("ModelTrainer initialized")

    def set_tickers(self, tickers: list):
        """Validate and set tickers"""
        try:
            if not isinstance(tickers, list):
                raise TypeError("Tickers must be a list")
            self.tickers = tickers
        except Exception as e:
            logging.error(f"Invalid tickers: {str(e)}")
            raise CustomException(e, sys) from e

    def load_data(self, ticker):
        """Load datasets for a specific ticker"""
        try:
            logging.info(f"Loading data for {ticker}")
            datasets = {}
            
            # Load train and test sets from the train_test_splits folder
            datasets['X_train'] = pd.read_csv(f'artifacts/train_test_splits/{ticker}_X_train.csv', index_col=0, parse_dates=True)
            datasets['X_test'] = pd.read_csv(f'artifacts/train_test_splits/{ticker}_X_test.csv', index_col=0, parse_dates=True)
            datasets['y_train'] = pd.read_csv(f'artifacts/train_test_splits/{ticker}_y_train.csv', index_col=0, parse_dates=True).squeeze("columns")
            datasets['y_test'] = pd.read_csv(f'artifacts/train_test_splits/{ticker}_y_test.csv', index_col=0, parse_dates=True).squeeze("columns")
            datasets['X_train_scaled'] = pd.read_csv(f'artifacts/train_test_splits/{ticker}_X_train_scaled.csv', index_col=0, parse_dates=True)
            datasets['X_test_scaled'] = pd.read_csv(f'artifacts/train_test_splits/{ticker}_X_test_scaled.csv', index_col=0, parse_dates=True)
            
            # Load full engineered data for time series analysis from the engineered_data folder
            datasets['full_data'] = pd.read_csv(f'artifacts/engineered_data/{ticker}_engineered.csv', index_col=0, parse_dates=True)
            
            # Load scaler
            with open(f'artifacts/train_test_splits/{ticker}_scaler.pkl', 'rb') as f:
                datasets['scaler'] = pickle.load(f)
            
            logging.info(f"Successfully loaded data for {ticker}")
            return datasets
            
        except Exception as e:
            logging.error(f"Error loading data for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def evaluate_model(self, y_true, y_pred, model_name):
        """Calculate and return regression evaluation metrics"""
        try:
            if y_true is None or y_pred is None:
                logging.warning(f"Missing input data for {model_name} evaluation")
                return {
                    'Model': model_name,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MAPE': np.nan,
                    'R2': np.nan
                }
            
            logging.info(f"\nDebug information for {model_name}:")
            logging.info(f"y_true shape: {y_true.shape if hasattr(y_true, 'shape') else 'no shape'}")
            logging.info(f"y_pred shape: {y_pred.shape if hasattr(y_pred, 'shape') else 'no shape'}")
            logging.info(f"NaN in y_true: {np.isnan(y_true).sum() if hasattr(y_true, 'sum') else 'unknown'}")
            logging.info(f"NaN in y_pred: {np.isnan(y_pred).sum() if hasattr(y_pred, 'sum') else 'unknown'}")
            
            if not isinstance(y_true, pd.Series):
                y_true = pd.Series(y_true)
            if not isinstance(y_pred, pd.Series):
                y_pred = pd.Series(y_pred, index=y_true.index)
            
            y_true, y_pred = y_true.align(y_pred, join='inner')
            
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) < 2:
                logging.warning(f"Not enough valid data points for {model_name} evaluation")
                logging.info(f"Valid points: {len(y_true_clean)} out of {len(y_true)}")
                return {
                    'Model': model_name,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MAPE': np.nan,
                    'R2': np.nan
                }
            
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            epsilon = 1e-10
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + epsilon))) * 100
            r2 = r2_score(y_true_clean, y_pred_clean)
            
            logging.info(f"Successfully calculated metrics for {model_name}")
            logging.info(f"Used {len(y_true_clean)} valid data points out of {len(y_true)}")
            return {
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            }
            
        except Exception as e:
            logging.error(f"Error calculating metrics for {model_name}: {str(e)}")
            return {
                'Model': model_name,
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan,
                'R2': np.nan,
                'Error': str(e)
            }

    def train_arima_model(self, ticker, data_dict):
        """Train ARIMA model using improved auto_arima"""
        try:
            logging.info(f"Starting ARIMA training for {ticker}")
            
            # Extract appropriate data
            if 'full_data' in data_dict and 'Close' in data_dict['full_data'].columns:
                close_prices = data_dict['full_data']['Close']
            elif 'y_train' in data_dict and 'y_test' in data_dict:
                close_prices = pd.concat([data_dict['y_train'], data_dict['y_test']])
            else:
                raise CustomException("Could not find suitable price data", sys)
            
            # Clean the data
            close_prices = pd.to_numeric(close_prices, errors='coerce')
            close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
            
            logging.info(f"\nClose prices summary:")
            logging.info(f"Length: {len(close_prices)}")
            logging.info(f"Range: {close_prices.min():.2f} to {close_prices.max():.2f}")
            logging.info(f"Mean: {close_prices.mean():.2f}")
            logging.info(f"First few values: {[round(x, 2) for x in close_prices[:5].tolist()]}")
            
            if 'y_train' in data_dict and 'y_test' in data_dict:
                # Use the existing train/test split
                train_data = data_dict['y_train']
                test_data = data_dict['y_test']
            else:
                # Create a new train/test split (80/20)
                train_size = int(len(close_prices) * 0.8)
                train_data = close_prices[:train_size]
                test_data = close_prices[train_size:]
            
            logging.info(f"Training data: {len(train_data)} points")
            logging.info(f"Testing data: {len(test_data)} points\n")
            
            # Run auto_arima
            best_model, best_order = self.auto_arima(
                train_data, 
                max_p=3, 
                max_d=2, 
                max_q=3,
                seasonal=False,
                max_tries=20,
                timeout=120
            )
            
            if best_model is None:
                logging.warning("AutoARIMA failed to find suitable model")
                
                try:
                    # Use a simple differencing approach
                    diff_series = train_data.diff().dropna()
                    mean_diff = diff_series.mean()

                    # Make forecasts
                    last_value = train_data.iloc[-1]
                    forecasts = [last_value]
                    for _ in range(len(test_data)):
                        next_value = forecasts[-1] + mean_diff
                        forecasts.append(next_value)

                    forecasts = forecasts[1:] 
                    forecasts = pd.Series(forecasts, index=test_data.index)

                    # Evaluate
                    metrics = self.evaluate_model(test_data, forecasts, model_name=f"ARIMA{best_order}")

                    return {
                        'metrics': metrics,
                        'predictions': forecasts,
                        'model_type': 'Mean Differencing'
                    }

                except Exception as e:
                    logging.warning(f"Fallback method also failed: {str(e)}")
                    return None
            
            logging.info("\nGenerating forecasts...")

            try:
                # Use one-step ahead forecasting for better accuracy
                history = train_data.tolist()
                predictions = []

                for t in range(len(test_data)):
                    if t % 20 == 0:
                        logging.info(f"Forecasting step {t+1}/{len(test_data)}")

                    try:
                        # Fit model
                        if len(best_order) <= 3:  # ARIMA
                            model = ARIMA(history, order=best_order)
                            model_fit = model.fit()
                        else:  # SARIMA
                            # Extract SARIMA components
                            p, d, q, P, D, Q, m = best_order
                            model = SARIMAX(history, order=(p,d,q), seasonal_order=(P,D,Q,m))
                            model_fit = model.fit(disp=False)

                        # Make one-step forecast
                        forecast = model_fit.forecast(steps=1)[0]
                        predictions.append(forecast)

                        # Add actual value to history
                        history.append(test_data.iloc[t])

                    except Exception as e:
                        logging.warning(f"Error at step {t}: {str(e)}")
                        # Use last prediction or mean
                        if len(predictions) > 0:
                            predictions.append(predictions[-1])
                        else:
                            predictions.append(train_data.mean())
            except Exception as e:
                raise CustomException(e, sys)
        
            predictions = pd.Series(predictions, index=test_data.index)

            # Handle any NaN values
            predictions = pd.Series(predictions, index=test_data.index)

            # Evaluate
            metrics = self.evaluate_model(test_data, predictions, model_name=f"ARIMA{best_order}")
        
            # Save model
            model_path = os.path.join("artifacts", f"{ticker}_arima_model.pkl")
            save_object(model_path, best_model)
            
            # logging results
            logging.info("\nARIMA Model Evaluation:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(value):
                    logging.info(f"{key}: {value:.4f}")
                else:
                    logging.info(f"{key}: {value}")
            
            logging.info(f"Successfully trained ARIMA model for {ticker}")
            return {
                'metrics': metrics,
                'model': best_model,
                'predictions': predictions,
                'model_order': best_order
            }
            
        except Exception as e:
            logging.error(f"Error in ARIMA training for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def train_sarimax_model(self, ticker, data_dict):
        try:
            """Train SARIMAX model using improved auto_arima"""
            logging.info(f"\n=== SARIMAX Modeling for {ticker} ===\n")
            
            train_data = data_dict['y_train']
            test_data = data_dict['y_test']
            
            # Extract appropriate data from the given structure
            if 'full_data' in data_dict:
                # Use full data if available
                logging.info("Using full_data for time series modeling")
                if isinstance(data_dict['full_data'], pd.DataFrame) and 'Close' in data_dict['full_data'].columns:
                    close_prices = data_dict['full_data']['Close']
                else:
                    logging.warning("Error: full_data doesn't have a 'Close' column")
                    return None
            elif 'y_train' in data_dict and 'y_test' in data_dict:
                # If separate train/test target variables are available, combine them
                logging.info("Using y_train and y_test for time series modeling")
                y_train = data_dict['y_train']
                y_test = data_dict['y_test']
                
                # Check if these are the closing prices or next day's prices
                if isinstance(y_train, pd.Series) and y_train.name == 'Next_Close':
                    logging.info("Note: Using 'Next_Close' as the target variable")
                
                # Combine train and test
                close_prices = pd.concat([y_train, y_test])
            else:
                logging.warning("Error: Could not find suitable price data in the provided structure")
                logging.info("Available keys:", list(data_dict.keys()))
                return None
            
            # Ensure close_prices is a Series and handle missing values
            if not isinstance(close_prices, pd.Series):
                try:
                    close_prices = pd.Series(close_prices)
                except Exception as e:
                    logging.warning(f"Error converting close prices to Series: {str(e)}")
                    return None
            
            # Clean the data
            close_prices = pd.to_numeric(close_prices, errors='coerce')
            close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
            
            # Print data summary
            logging.info(f"\nClose prices summary:")
            logging.info(f"Length: {len(close_prices)}")
            logging.info(f"Range: {close_prices.min():.2f} to {close_prices.max():.2f}")
            logging.info(f"Mean: {close_prices.mean():.2f}")
            logging.info(f"First few values: {[round(x, 2) for x in close_prices[:5].tolist()]}")
            
            # Split the data (if not already split)
            if 'y_train' in data_dict and 'y_test' in data_dict:
                # Use the existing train/test split
                train_data = data_dict['y_train']
                test_data = data_dict['y_test']
            else:
                # Create a new train/test split (80/20)
                train_size = int(len(close_prices) * 0.8)
                train_data = close_prices[:train_size]
                test_data = close_prices[train_size:]
            
            logging.info(f"Training data: {len(train_data)} points")
            logging.info(f"Testing data: {len(test_data)} points\n")
            
            # Determine appropriate seasonal period based on data frequency
            if isinstance(train_data.index, pd.DatetimeIndex):
                # For daily data, use 5 for business week
                # For monthly data, use 12 for annual cycle
                if train_data.index.freq == 'D' or train_data.index.freq == 'B':
                    m = 5  # Business week
                elif train_data.index.freq == 'M':
                    m = 12  # Annual cycle
                else:
                    # Try to infer frequency
                    try:
                        inferred_freq = pd.infer_freq(train_data.index)
                        if inferred_freq in ['D', 'B']:
                            m = 5
                        elif inferred_freq == 'M':
                            m = 12
                        else:
                            m = 5  # Default for financial data
                    except:
                        m = 5 
            else:
                m = 5  
            
            logging.info(f"Using seasonal period m={m}")
            
            # Run auto_arima with seasonal components
            logging.info("Running AutoARIMA for SARIMAX...")
            best_model, best_order = self.auto_arima(
                train_data, 
                max_p=2, 
                max_d=1, 
                max_q=2,
                seasonal=True,
                max_P=1,
                max_D=1,
                max_Q=1,
                m=m,
                max_tries=20,
                timeout=120
            )
            
            if best_model is None or len(best_order) <= 3:  # Not a seasonal model
                logging.warning("AutoARIMA failed to find suitable seasonal model")
                # Fallback to a simple seasonal model
                try:
                    # Try a simple (1,1,1)x(1,1,1,m) model
                    logging.info("Trying manual SARIMAX(1,1,1)x(1,1,1,m) model")
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, m)
                    
                    model = SARIMAX(train_data, 
                                order=order, 
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                    
                    model_fit = model.fit(disp=False, method='powell')
                    
                    # Make predictions using one-step forecasting
                    history = train_data.tolist()
                    predictions = []
                    
                    for t in range(len(test_data)):
                        if t % 20 == 0:
                            logging.info(f"Forecasting step {t+1}/{len(test_data)}")
                        
                        try:
                            # Fit model
                            model = SARIMAX(history, 
                                        order=order, 
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                            model_fit = model.fit(disp=False)
                            
                            # Make one-step forecast
                            forecast = model_fit.forecast(steps=1)[0]
                            predictions.append(forecast)
                            
                            # Add actual value to history
                            history.append(test_data.iloc[t])
                            
                        except Exception as e:
                            logging.warning(f"Error at step {t}: {str(e)}")
                            # Use last prediction or mean
                            if len(predictions) > 0:
                                predictions.append(predictions[-1])
                            else:
                                predictions.append(train_data.mean())
                    
                    # Convert to Series
                    predictions = pd.Series(predictions, index=test_data.index)
                    
                    # Evaluate
                    metrics = self.evaluate_model(test_data, predictions, model_name=f"SARIMAX{order}x{seasonal_order}")
                    
                    best_order = (*order, *seasonal_order)
                    
                except Exception as e:
                    logging.info(f"Manual SARIMAX also failed: {str(e)}")
                    return None
            else:
                # Generate forecasts using the best model
                logging.info("\nGenerating forecasts...")
                
                try:
                    # Extract components
                    if len(best_order) > 3:
                        p, d, q, P, D, Q, m = best_order
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, m)
                    else:
                        order = best_order
                        seasonal_order = (0, 0, 0, m)  # No seasonal component
                    
                    # Use one-step ahead forecasting
                    history = train_data.tolist()
                    predictions = []
                    
                    for t in range(len(test_data)):
                        if t % 20 == 0:
                            logging.info(f"Forecasting step {t+1}/{len(test_data)}")
                        
                        try:
                            # Fit model
                            model = SARIMAX(history, 
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                            model_fit = model.fit(disp=False)
                            
                            # Make one-step forecast
                            forecast = model_fit.forecast(steps=1)[0]
                            predictions.append(forecast)
                            
                            # Add actual value to history
                            history.append(test_data.iloc[t])
                            
                        except Exception as e:
                            logging.warning(f"Error at step {t}: {str(e)}")
                            # Use last prediction or mean
                            if len(predictions) > 0:
                                predictions.append(predictions[-1])
                            else:
                                predictions.append(train_data.mean())
                    
                    # Convert to Series
                    predictions = pd.Series(predictions, index=test_data.index)
                    
                    # Handle any NaN values
                    predictions = predictions.fillna(method='ffill').fillna(method='bfill')
                    
                    # Evaluate
                    metrics = self.evaluate_model(test_data, predictions, model_name=f"SARIMAX{order}x{seasonal_order}")
                
                except Exception as e:
                    logging.warning(f"Error generating forecasts: {str(e)}")
                    return None
            
            # Output results
            logging.info("\nSARIMAX Model Evaluation:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(value):
                    logging.info(f"{key}: {value:.4f}")
                else:
                    logging.info(f"{key}: {value}")
            
            
            return {
                'metrics': metrics,
                'model': model_fit, 
                'predictions': predictions,
                'model_order': best_order
            }
        except Exception as e:
            logging.error(f"Error in SARIMAX training for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def train_gradient_boosting(self, ticker, data_dict):
        """Train Gradient Boosting Regressor model"""
        try:
            logging.info(f"Starting Gradient Boosting training for {ticker}")
            start_time = time.time()
            
            # Extract features and target
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
            y_train = data_dict['y_train']
            y_test = data_dict['y_test']
            
            logging.info(f"Training features shape: {X_train.shape}")
            logging.info(f"Testing features shape: {X_test.shape}")
            logging.info(f"Training target shape: {y_train.shape}")
            logging.info(f"Testing target shape: {y_test.shape}")

            # Check for any NaN values
            logging.info(f"NaN in X_train: {X_train.isna().sum().sum()}")
            logging.info(f"NaN in X_test: {X_test.isna().sum().sum()}")
            logging.info(f"NaN in y_train: {y_train.isna().sum()}")
            logging.info(f"NaN in y_test: {y_test.isna().sum()}")
            
            # Fill any missing values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_test.mean())
            
            # Initialize and train the model
            logging.info("\nTraining Gradient Boosting model...")
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            
            gb_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = gb_model.predict(X_train)
            y_pred_test = gb_model.predict(X_test)

            # Evaluate model
            train_metrics = self.evaluate_model(y_train, y_pred_train, model_name="GB-Train")
            test_metrics = self.evaluate_model(y_test, y_pred_test, model_name="GB-Test")
            
            training_time = time.time() - start_time
            
    
            logging.info("\nGradient Boosting Train Evaluation:")
            for metric, value in train_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.info(f"{metric}: {value}")

            logging.info("\nGradient Boosting Test Evaluation:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.info(f"{metric}: {value}")

            logging.info(f"\nTraining time: {training_time:.2f} seconds")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': gb_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            logging.info("\nTop 10 Most Important Features:")
            logging.info(feature_importance.head(10).to_string(index=False))
            
            # Save model
            model_path = os.path.join("artifacts", f"{ticker}_gb_model.pkl")
            save_object(model_path, gb_model)
            
            logging.info(f"Successfully trained Gradient Boosting model for {ticker}")
            return {
                'test_metrics': test_metrics,
                'model': gb_model,
                'predictions': y_pred_test,
                'feature_importance': feature_importance,
                'training_time': time.time() - start_time
            }
            
        except Exception as e:
            logging.error(f"Error in Gradient Boosting training for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def train_xgboost_model(self, ticker, data_dict):
        """Train XGBoost model"""
        try:
            logging.info(f"Starting XGBoost training for {ticker}")
            start_time = time.time()
            
            # Extract features and target
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
            y_train = data_dict['y_train']
            y_test = data_dict['y_test']
            
            logging.info(f"Training features shape: {X_train.shape}")
            logging.info(f"Testing features shape: {X_test.shape}")
            logging.info(f"Training target shape: {y_train.shape}")
            logging.info(f"Testing target shape: {y_test.shape}")
            
            logging.info(f"NaN in X_train: {X_train.isna().sum().sum()}")
            logging.info(f"NaN in X_test: {X_test.isna().sum().sum()}")
            logging.info(f"NaN in y_train: {y_train.isna().sum()}")
            logging.info(f"NaN in y_test: {y_test.isna().sum()}")
            
            # Fill any missing values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_test.mean())
            
            # Initialize and train the model
            logging.info("\nTraining XGBoost model...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            try:    
                logging.info("Fitting XGBoost model...")
                xgb_model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    early_stopping_rounds=20,
                    verbose=False
                )

                logging.info("XGBoost model fitted successfully")
            except Exception as e:

                logging.warning(f"First fit attempt failed: {str(e)}")
                # Alternative approach for older XGBoost versions
                try:
                    # Define evaluation metrics separately
                    eval_metric = ['rmse']

                    # Try fitting with eval_metric in the constructor
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        min_child_weight=2,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='reg:squarederror',
                        eval_metric=eval_metric,  # Include in constructor
                        random_state=42,
                        n_jobs=-1
                    )

                    # Fit without eval_metric parameter
                    xgb_model.fit(
                        X_train, 
                        y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        early_stopping_rounds=20,
                        verbose=False
                    )

                    logging.info("XGBoost model fitted successfully with alternative approach")
                except Exception as e2:
                    logging.warning(f"Second fit attempt also failed: {str(e2)}")

                    # Last resort - basic fit without evaluation
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        min_child_weight=2,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1
                    )

                    xgb_model.fit(X_train, y_train)
                    logging.info("XGBoost model fitted with basic approach (no early stopping)")
            
            best_iteration = None
            if hasattr(xgb_model, 'best_iteration'):
                best_iteration = xgb_model.best_iteration
            elif hasattr(xgb_model, 'best_ntree_limit'):
                best_iteration = xgb_model.best_ntree_limit

            if best_iteration:
                logging.info(f"Best iteration: {best_iteration}")
            
            # Make predictions
            y_pred_train = xgb_model.predict(X_train)
            y_pred_test = xgb_model.predict(X_test)

            # Evaluate model
            train_metrics = self.evaluate_model(y_train, y_pred_train,model_name="XGB-Train")
            test_metrics = self.evaluate_model(y_test, y_pred_test, model_name="XGB-Test")
            
            training_time = time.time() - start_time
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Save model
            model_path = os.path.join("artifacts", f"{ticker}_xgb_model.pkl")
            save_object(model_path, xgb_model)
            
            logging.info(f"Successfully trained XGBoost model for {ticker}")
            
            # Print evaluation results
            logging.info("\nXGBoost Train Evaluation:")
            for metric, value in train_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.info(f"{metric}: {value}")

            logging.info("\nXGBoost Test Evaluation:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.info(f"{metric}: {value}")

            logging.info(f"\nTraining time: {training_time:.2f} seconds")
            
            # Feature importance analysis
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            logging.info("\nTop 10 Most Important Features:")
            logging.info(feature_importance.head(10).to_string(index=False))
            
            return {
                'test_metrics': test_metrics,
                'model': xgb_model,
                'predictions': y_pred_test,
                'feature_importance': feature_importance,
                'training_time': time.time() - start_time
            }
            
        except Exception as e:
            logging.error(f"Error in XGBoost training for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def train_lightgbm_model(self, ticker, data_dict):
        """Train LightGBM model"""
        try:
            logging.info(f"Starting LightGBM training for {ticker}")
            start_time = time.time()
            
            # Extract features and target
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
            y_train = data_dict['y_train']
            y_test = data_dict['y_test']
            
            logging.info(f"Training features shape: {X_train.shape}")
            logging.info(f"Testing features shape: {X_test.shape}")
            logging.info(f"Training target shape: {y_train.shape}")
            logging.info(f"Testing target shape: {y_test.shape}")
            
            logging.info(f"NaN in X_train: {X_train.isna().sum().sum()}")
            logging.info(f"NaN in X_test: {X_test.isna().sum().sum()}")
            logging.info(f"NaN in y_train: {y_train.isna().sum()}")
            logging.info(f"NaN in y_test: {y_test.isna().sum()}")
            
            # Fill any missing values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_test.mean())
        
            logging.info("\nTraining LightGBM model...")
            try:    
                # Initialize and train the model
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='regression',
                    random_state=42,
                    n_jobs=-1
                )

                lgb_model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    early_stopping_rounds=20,
                    verbose=False
                )

                logging.info("LightGBM model fitted successfully")
            except Exception as e:
                logging.info(f"First fit attempt failed: {str(e)}")

                try:
                    # Try alternative approach
                    lgb_model = lgb.LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        min_child_samples=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='regression',
                        metric='rmse',
                        random_state=42,
                        n_jobs=-1
                    )

                    lgb_model.fit(
                        X_train, 
                        y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        early_stopping_rounds=20,
                        verbose=False
                    )

                    logging.info("LightGBM model fitted successfully with alternative approach")
                except Exception as e2:
                    logging.info(f"Second fit attempt also failed: {str(e2)}")
                    # Last resort - basic fit
                    lgb_model = lgb.LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        min_child_samples=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='regression',
                        random_state=42,
                        n_jobs=-1
                    )

                    lgb_model.fit(X_train, y_train)
                    logging.info("LightGBM model fitted with basic approach (no early stopping)")

                # Get best iteration if available
            best_iteration = None
            if hasattr(lgb_model, 'best_iteration_'):
                best_iteration = lgb_model.best_iteration_

            if best_iteration:
                logging.info(f"Best iteration: {best_iteration}")
            
            # Make predictions
            y_pred_train = lgb_model.predict(X_train)
            y_pred_test = lgb_model.predict(X_test)

            # Evaluate model
            train_metrics = self.evaluate_model(y_train, y_pred_train, model_name="LightGBM-Train")
            test_metrics = self.evaluate_model(y_test, y_pred_test, model_name="LightGBM-Test")
            
            training_time = time.time() - start_time
            
            # log evaluation results
            logging.info("\nLightGBM Train Evaluation:")
            for metric, value in train_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.info(f"{metric}: {value}")

            logging.info("\nLightGBM Test Evaluation:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.info(f"{metric}: {value}")

            logging.info(f"\nTraining time: {training_time:.2f} seconds")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': lgb_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            logging.info("\nTop 10 Most Important Features:")
            logging.info(feature_importance.head(10).to_string(index=False))
            
            # Save model
            model_path = os.path.join("artifacts", f"{ticker}_lgb_model.pkl")
            save_object(model_path, lgb_model)
            
            logging.info(f"Successfully trained LightGBM model for {ticker}")
            return {
                'test_metrics': test_metrics,
                'model': lgb_model,
                'predictions': y_pred_test,
                'feature_importance': feature_importance,
                'training_time': time.time() - start_time
            }
            
        except Exception as e:
            logging.error(f"Error in LightGBM training for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def auto_arima(self, time_series, max_p=3, max_d=2, max_q=3, seasonal=False, 
                   max_P=2, max_D=1, max_Q=2, m=12, information_criterion='aic', 
                   max_order=None, max_tries=50, timeout=60):
        """Implementation of auto_arima that is more robust to challenging datasets"""
        try:
            # Ensure time_series is a pandas Series
            if not isinstance(time_series, pd.Series):
                time_series = pd.Series(time_series)
            
            # Handle missing values
            time_series = time_series.fillna(method='ffill').fillna(method='bfill')
            
            # Check for constant series
            if time_series.std() == 0:
                logging.warning("Constant time series. Cannot fit ARIMA model.")
                return None, (0, 0, 0)
            
            # Check minimum length
            if len(time_series) < 10:
                logging.warning("Time series too short. Need at least 10 observations.")
                return None, (0, 0, 0)
            
            logging.info("Checking stationarity...")
            
            # Determine reasonable 'd' parameter using stationarity tests
            d_values = list(range(max_d + 1))
            adf_result = adfuller(time_series, regression='c')
            adf_stationary = adf_result[1] < 0.05
            
            if adf_stationary:
                d_values = [0] + d_values[1:]
                logging.info("Series is stationary according to ADF test.")
            else:
                logging.info("Series is not stationary according to ADF test.")
                
                try:
                    kpss_result = kpss(time_series, regression='c')
                    kpss_stationary = kpss_result[1] > 0.05
                    
                    if not kpss_stationary:
                        logging.info("KPSS test confirms non-stationarity.")
                        # Prioritize d=1 since both tests indicate non-stationarity
                        if 1 in d_values:
                            d_values.remove(1)
                            d_values = [1] + d_values
                    else:
                        logging.info("KPSS test suggests stationarity. Results conflict with ADF.")
                except:
                    logging.warning("KPSS test failed. Continuing with ADF results.")
            
            # Determine seasonal D if applicable
            D_values = list(range(max_D + 1))
            if seasonal and len(time_series) > m * 2:
                # Check seasonal stationarity by looking at seasonal differences
                seasonal_diff = time_series.diff(m).dropna()
                if len(seasonal_diff) > 10:
                    try:
                        seasonal_adf = adfuller(seasonal_diff, regression='c')
                        if seasonal_adf[1] < 0.05:
                            # Seasonal differencing made it stationary
                            if 1 in D_values:
                                D_values.remove(1)
                                D_values = [1] + D_values
                    except:
                        logging.warning("Seasonal stationarity test failed. Using default D values.")
            # Function to evaluate an ARIMA model with error handling
            def evaluate_arima(order, seasonal_order=None):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        if seasonal and seasonal_order is not None:
                            model = SARIMAX(time_series, 
                                           order=order, 
                                           seasonal_order=seasonal_order,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
                        else:
                            model = ARIMA(time_series, order=order)
                        
                        model_fit = model.fit(method='powell', disp=False, maxiter=100)
                        
                        if information_criterion.lower() == 'aic':
                            ic = model_fit.aic
                        elif information_criterion.lower() == 'bic':
                            ic = model_fit.bic
                        elif information_criterion.lower() == 'hqic':
                            ic = model_fit.hqic
                        else:
                            ic = model_fit.aic
                        
                        return ic, model_fit
                        
                except Exception as e:
                    return float('inf'), None
            
            # Generate parameter combinations
            p_values = range(max_p + 1)
            q_values = range(max_q + 1)
            parameter_combinations = []
            
            if seasonal:
                P_values = range(max_P + 1)
                Q_values = range(max_Q + 1)
                
                for p, d, q in product(p_values, d_values, q_values):
                    if max_order is not None and p + q > max_order:
                        continue
                    parameter_combinations.append((p, d, q, None))
                
                for p, d, q, P, D, Q in product(p_values, d_values, q_values, P_values, D_values, Q_values):
                    if max_order is not None and p + q + P + Q > max_order:
                        continue
                    if P == 0 and D == 0 and Q == 0:
                        continue
                    parameter_combinations.append((p, d, q, (P, D, Q, m)))
            else:
                for p, d, q in product(p_values, d_values, q_values):
                    if max_order is not None and p + q > max_order:
                        continue
                    parameter_combinations.append((p, d, q, None))
            
            # Limit number of combinations to max_tries
            if max_tries < len(parameter_combinations):
                prioritized = []
                for order in [(1,1,1), (0,1,1), (1,1,0), (2,1,2)]:
                    if order[1] in d_values and order in [(p,d,q) for p,d,q,_ in parameter_combinations]:
                        prioritized.append((order[0], order[1], order[2], None))
                
                remaining = [comb for comb in parameter_combinations if comb not in prioritized]
                np.random.shuffle(remaining)
                parameter_combinations = prioritized + remaining
                parameter_combinations = parameter_combinations[:max_tries]
            
            logging.info(f"Testing {len(parameter_combinations)} parameter combinations...")
            
            best_ic = float('inf')
            best_model = None
            best_config = None
            start_time = time.time()
            
            for i, (p, d, q, seasonal_order) in enumerate(parameter_combinations):
                if time.time() - start_time > timeout:
                    logging.warning(f"Timeout reached after testing {i} combinations.")
                    break
                
                # Report progress
                if (i+1) % 5 == 0 or i == 0:
                    msg = f"Testing combination {i+1}/{len(parameter_combinations)}: "
                    if seasonal_order:
                        msg += f"SARIMA({p},{d},{q})x{seasonal_order}..."
                    else:
                        msg += f"ARIMA({p},{d},{q})..."
                    logging.info(msg)
                
                ic, model = evaluate_arima((p, d, q), seasonal_order)
                
                if model is not None and np.isfinite(ic):
                    if ic < best_ic:
                        best_ic = ic
                        best_model = model
                        best_config = (p, d, q) if seasonal_order is None else (p, d, q, *seasonal_order)
            
            if best_model is None:
                logging.warning("No suitable model found. Trying basic models...")
                
                fallback_orders = [(0,1,0), (1,1,0), (0,1,1)]
                for order in fallback_orders:
                    logging.info(f"Trying fallback ARIMA{order}...")
                    ic, model = evaluate_arima(order)
                    if model is not None and np.isfinite(ic):
                        best_model = model
                        best_config = order
                        break
                    else:
                        logging.info("Failed.")
            
            if best_model is None:
                logging.error("All ARIMA models failed.")
            else:
                logging.info(f"Best model: {'SARIMA' if len(best_config) > 3 else 'ARIMA'}{best_config}")
                logging.info(f"Best {information_criterion.upper()}: {best_ic:.2f}")
            
            return best_model, best_config
            
        except Exception as e:
            logging.error(f"Error in auto_arima: {str(e)}")
            raise CustomException(e, sys)

    def compare_all_models(self, ticker, results):
        """Compare all trained models for a ticker"""
        try:
            logging.info(f"\n=== Model Comparison for {ticker} ===\n")
            
            # Gather test metrics
            models = [
                {
                    'Model': 'ARIMA', 
                    'RMSE': results['arima']['metrics']['RMSE'] if 'arima' in results and 'metrics' in results['arima'] else np.nan,
                    'Type': 'Time Series',
                    'MAE': np.nan,  # ARIMA doesn't compute MAE in this setup
                    'MAPE': np.nan,
                    'R2': np.nan,
                    'Time': np.nan
                },
                {
                    'Model': 'SARIMAX', 
                    'RMSE': results['sarimax']['metrics']['RMSE'] if 'sarimax' in results and 'metrics' in results['sarimax'] else np.nan,
                    'Type': 'Time Series',
                    'MAE': results['sarimax']['metrics']['MAE'] if 'metrics' in results['sarimax'] else np.nan,
                    'MAPE': results['sarimax']['metrics']['MAPE'] if 'metrics' in results['sarimax'] else np.nan,
                    'R2': results['sarimax']['metrics']['R2'] if 'metrics' in results['sarimax'] else np.nan,
                    'Time': np.nan  # SARIMAX doesn't track training time here
                },
                {
                    'Model': 'Gradient Boosting', 
                    'RMSE': results['gb']['test_metrics']['RMSE'] if 'gb' in results and 'test_metrics' in results['gb'] else np.nan, 
                    'Type': 'Tree-based', 
                    'MAE': results['gb']['test_metrics']['MAE'] if 'test_metrics' in results['gb'] else np.nan, 
                    'MAPE': results['gb']['test_metrics']['MAPE'] if 'test_metrics' in results['gb'] else np.nan, 
                    'R2': results['gb']['test_metrics']['R2'] if 'test_metrics' in results['gb'] else np.nan, 
                    'Time': results['gb']['training_time'] if 'training_time' in results['gb'] else np.nan
                },
                {
                    'Model': 'XGBoost', 
                    'RMSE': results['xgb']['test_metrics']['RMSE'] if 'xgb' in results and 'test_metrics' in results['xgb'] else np.nan, 
                    'Type': 'Tree-based', 
                    'MAE': results['xgb']['test_metrics']['MAE'] if 'test_metrics' in results['xgb'] else np.nan, 
                    'MAPE': results['xgb']['test_metrics']['MAPE'] if 'test_metrics' in results['xgb'] else np.nan, 
                    'R2': results['xgb']['test_metrics']['R2'] if 'test_metrics' in results['xgb'] else np.nan, 
                    'Time': results['xgb']['training_time'] if 'training_time' in results['xgb'] else np.nan
                },
                {
                    'Model': 'LightGBM', 
                    'RMSE': results['lgb']['test_metrics']['RMSE'] if 'lgb' in results and 'test_metrics' in results['lgb'] else np.nan, 
                    'Type': 'Tree-based', 
                    'MAE': results['lgb']['test_metrics']['MAE'] if 'test_metrics' in results['lgb'] else np.nan, 
                    'MAPE': results['lgb']['test_metrics']['MAPE'] if 'test_metrics' in results['lgb'] else np.nan, 
                    'R2': results['lgb']['test_metrics']['R2'] if 'test_metrics' in results['lgb'] else np.nan, 
                    'Time': results['lgb']['training_time'] if 'training_time' in results['lgb'] else np.nan
                }
            ]
            
            # Create DataFrame
            comparison_df = pd.DataFrame(models)
            
            # Sort by RMSE (ascending)
            comparison_df = comparison_df.sort_values('RMSE')
            
            # Print comparison table
            logging.info("Model Performance Comparison:")
            logging.info(comparison_df.to_string(index=False))
            
            # Find best model
            best_model = comparison_df.iloc[0]['Model']
            logging.info(f"\nBest model: {best_model} with RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
            
            
            return comparison_df
        
        except Exception as e:
            logging.error(f"Error comparing models for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_training(self, ticker, artifacts_path='artifacts'):
        """Run all models, save the results, and save trained models to pickle files."""
        try:
            logging.info(f"\nLoading data for {ticker}...")
            
            # Load data from artifacts directory
            data_dict = self.load_data(ticker)
            
            results = {}
            
            # Train ARIMA model
            logging.info("\nTraining ARIMA model...")
            arima_results = self.train_arima_model(ticker, data_dict)
            results['arima'] = arima_results
            
            # Train SARIMAX model
            logging.info("\nTraining SARIMAX model...")
            sarimax_results = self.train_sarimax_model(ticker, data_dict)
            results['sarimax'] = sarimax_results
            
            # Train Gradient Boosting
            logging.info("\nTraining Gradient Boosting model...")
            gb_results = self.train_gradient_boosting(ticker, data_dict)
            results['gb'] = gb_results
            
            # Train XGBoost
            logging.info("\nTraining XGBoost model...")
            xgb_results = self.train_xgboost_model(ticker, data_dict)
            results['xgb'] = xgb_results
            
            # Train LightGBM
            logging.info("\nTraining LightGBM model...")
            lgb_results = self.train_lightgbm_model(ticker, data_dict)
            results['lgb'] = lgb_results
            
            # Save all trained models
            saved_paths = self.save_models(ticker, results, artifacts_path)
            results['saved_model_paths'] = saved_paths
            
            # Save metrics to training_results.pkl
            training_results_path = os.path.join(artifacts_path, "training_results.pkl")
            try:
                if os.path.exists(training_results_path):
                    with open(training_results_path, 'rb') as f:
                        training_results = pickle.load(f)
                else:
                    training_results = {}
                
                training_results[ticker] = {
                    'arima': arima_results.get('metrics', {}),
                    'sarimax': sarimax_results.get('metrics', {}),
                    'gb': gb_results.get('test_metrics', {}),
                    'xgb': xgb_results.get('test_metrics', {}),
                    'lgb': lgb_results.get('test_metrics', {})
                }
                
                with open(training_results_path, 'wb') as f:
                    pickle.dump(training_results, f)
                logging.info(f"Saved training results to {training_results_path}")
            
            except Exception as e:
                logging.error(f"Error saving training results: {str(e)}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in model training initiation for {ticker}: {str(e)}")
            raise CustomException(e, sys)

    def save_models(self, ticker, results, artifacts_path='artifacts'):
        """Save trained models to pickle files"""
        
        # Fix artifacts path
        artifacts_path = artifacts_path.replace('\\', '/')
        
        logging.info(f"Saving models to: {artifacts_path}")
        
        # Create artifacts directory if it doesn't exist
        try:
            os.makedirs(artifacts_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directory: {str(e)}")
            return {}
        
        # Save each model
        models_to_save = {
            'arima': results.get('arima', {}).get('model'),
            'sarimax': results.get('sarimax', {}).get('model'),
            'gb': results.get('gb', {}).get('model'),
            'xgb': results.get('xgb', {}).get('model'),
            'lgb': results.get('lgb', {}).get('model')
        }
        
        # Also save SARIMAX model order if available
        if 'sarimax' in results and results['sarimax'] and 'model_order' in results['sarimax']:
            try:
                order_path = os.path.join(artifacts_path, f'{ticker}_sarimax_order.pkl')
                with open(order_path, 'wb') as f:
                    pickle.dump(results['sarimax']['model_order'], f)
                logging.info(f"Saved SARIMAX model order to {order_path}")
            except Exception as e:
                logging.error(f"Error saving SARIMAX model order: {str(e)}")
        
        saved_paths = {}
        for model_name, model in models_to_save.items():
            if model is not None:
                try:
                    model_path = os.path.join(artifacts_path, f'{ticker}_{model_name}_model.pkl')
                    model_path = model_path.replace('\\', '/')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    saved_paths[model_name] = model_path
                    logging.info(f"Saved {model_name} model to {model_path}")
                except Exception as e:
                    logging.error(f"Error saving {model_name} model: {str(e)}")
        
        return saved_paths