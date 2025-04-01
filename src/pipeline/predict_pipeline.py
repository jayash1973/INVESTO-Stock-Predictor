import os
import sys
import logging
import pandas as pd
from typing import Dict, Any
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self, ticker: str = "AAPL", forecast_days: int = 1):
        self.ticker = ticker
        self.forecast_days = forecast_days  # Store the forecast_days
        self.artifacts_dir = "artifacts"
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.engineered_data = None  # Store engineered data
        self.training_features = None  # Store feature names from training
        logging.info(f"Initialized PredictPipeline for {self.ticker} with forecast days: {self.forecast_days}")

    def load_artifacts(self):
        """Load all required artifacts for prediction"""
        try:
            # Load models
            model_types = ['arima', 'sarimax', 'gb', 'xgb', 'lgb']
            for model_type in model_types:
                model_path = os.path.join(
                    self.artifacts_dir,
                    f"{self.ticker}_{model_type}_model.pkl"
                )
                if os.path.exists(model_path):
                    self.models[model_type] = load_object(model_path)
                    logging.info(f"Loaded {model_type.upper()} model from {model_path}")
                else:
                    logging.warning(f"Model not found: {model_path}")

            # Load scaler and get feature names
            scaler_path = os.path.join(
                self.artifacts_dir,
                "train_test_splits",
                f"{self.ticker}_scaler.pkl"
            )
            self.scaler = load_object(scaler_path)
            if hasattr(self.scaler, 'feature_names_in_'):
                self.training_features = list(self.scaler.feature_names_in_)
            logging.info(f"Loaded scaler from {scaler_path}")

            # Load engineered data
            data_path = os.path.join(
                self.artifacts_dir,
                "engineered_data",
                f"{self.ticker}_engineered.csv"
            )
            self.engineered_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logging.info(f"Loaded engineered data from {data_path}")

        except Exception as e:
            logging.error(f"Error loading artifacts: {str(e)}")
            raise CustomException(e, sys)

    def prepare_features(self):
        """Prepare features for different model types"""
        try:
            if self.training_features is None:
                raise CustomException("Training feature names not available", sys)
            
            # Get the latest data point
            latest_data = self.engineered_data.iloc[[-1]].copy()
            
            # Ensure we only use features that were in training
            tree_features = latest_data[self.training_features]
            
            # Scale features and convert to DataFrame to retain column names
            scaled_features = pd.DataFrame(self.scaler.transform(tree_features), columns=self.training_features)
            
            # For time series models
            close_prices = self.engineered_data['Close']
            
            return {
                'tree_features': scaled_features,
                'time_series': close_prices,
                'latest_date': self.engineered_data.index[-1],
                'feature_names': self.training_features
            }
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            raise CustomException(e, sys)

    def predict(self) -> Dict[str, float]:
        """Make predictions using all loaded models"""
        try:
            self.load_artifacts()
            features = self.prepare_features()
            predictions = {}
            latest_date_str = features['latest_date'].strftime('%Y-%m-%d')

            # Time series models predictions
            if 'arima' in self.models:
                try:
                    arima_forecast = self.models['arima'].forecast(steps=self.forecast_days)  # Use forecast_days
                    predictions['ARIMA'] = {
                        'date': latest_date_str,
                        'prediction': round(arima_forecast[-1], 2)  # Get the last forecasted value
                    }
                except Exception as e:
                    logging.warning(f"ARIMA prediction failed: {str(e)}")

            if 'sarimax' in self.models:
                try:
                    sarimax_forecast = self.models['sarimax'].forecast(steps=self.forecast_days)  # Use forecast_days
                    predictions['SARIMAX'] = {
                        'date': latest_date_str,
                        'prediction': round(sarimax_forecast[-1], 2)  # Get the last forecasted value
                    }
                except Exception as e:
                    logging.warning(f"SARIMAX prediction failed: {str(e)}")

            # Tree-based models predictions
            for model_type in ['gb', 'xgb', 'lgb']:
                if model_type in self.models:
                    try:
                        model_name = model_type.upper()
                        pred = self.models[model_type].predict(features['tree_features'])
                        predictions[model_name] = {
                            'date': latest_date_str,
                            'prediction': round(pred[0], 2)  # Assuming single prediction for tree-based models
                        }
                    except Exception as e:
                        logging.warning(f"{model_name} prediction failed: {str(e)}")

            if not predictions:
                raise CustomException("All model predictions failed", sys)

            logging.info("Predictions completed successfully")
            return predictions

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise CustomException(e, sys)

class PredictionData:
    """Helper class to format prediction results"""
    def __init__(self, predictions: Dict[str, float]):
        self.predictions = predictions
    
    def get_formatted_predictions(self):
        """Format predictions for output"""
        try:
            formatted = {}
            for model, data in self.predictions.items():
                formatted[model] = {
                    'prediction_date': data['date'],
                    'predicted_close': data['prediction']
                }
            return formatted
        except Exception as e:
            logging.error(f"Formatting error: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Starting prediction pipeline")
        
        # Initialize and run pipeline
        pipeline = PredictPipeline(ticker="AAPL")
        raw_predictions = pipeline.predict()
        
        # Format results
        results = PredictionData(raw_predictions).get_formatted_predictions()
        
        logging.info("\nPrediction Results:")
        for model, prediction in results.items():
            logging.info(
                f"{model}: {prediction['predicted_close']} "
                f"(for {prediction['prediction_date']})"
            )
            
    except Exception as e:
        logging.error(f"Prediction pipeline failed: {str(e)}")