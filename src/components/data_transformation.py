import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Import custom exception and logging
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    engineered_data_dir: str = os.path.join('artifacts', 'engineered_data')
    train_test_split_dir: str = os.path.join('artifacts', 'train_test_splits')
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # Ensure directories exist
        os.makedirs(self.data_transformation_config.engineered_data_dir, exist_ok=True)
        os.makedirs(self.data_transformation_config.train_test_split_dir, exist_ok=True)
    
    def check_stationarity(self, series, window=12, title=''):
        """
        Check stationarity of a time series with Dickey-Fuller test
        
        Args:
            series: Pandas Series with time series data
            window: Rolling window size for mean/std calculations
            title: Title for the output
        """
        logging.info(f"Checking stationarity for {title}")
        
        # Handle empty or all-NA series
        if series.dropna().empty:
            logging.info(f"Cannot check stationarity for {title} - series is empty after dropping NA values")
            return
        
        # Rolling statistics
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        
        # Dickey-Fuller test
        try:
            result = adfuller(series.dropna())
            
            logging.info(f'Augmented Dickey-Fuller Test for {title}:')
            logging.info(f'ADF Statistic: {result[0]:.6f}')
            logging.info(f'p-value: {result[1]:.6f}')
            logging.info(f'Critical Values:')
            for key, value in result[4].items():
                logging.info(f'\t{key}: {value:.6f}')
            
            # Interpret results
            if result[1] <= 0.05:
                logging.info("Result: Series is STATIONARY (reject null hypothesis)")
            else:
                logging.info("Result: Series is NON-STATIONARY (fail to reject null hypothesis)")
        
        except Exception as e:
            logging.error(f"Error performing ADF test for {title}: {str(e)}")
            return
    
    def calculate_daily_returns(self, df):
        """Calculate daily returns if not present"""
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change() * 100
        return df
    
    def engineer_features(self, df, ticker):
        """
        Engineer features for the stock data
        
        Args:
            df: DataFrame with stock data
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with engineered features
        """
        logging.info(f"Engineering features for {ticker}...")
        df = df.copy()
        
        # 1. Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                error_msg = f"Required column {col} not found in dataframe"
                logging.error(error_msg)
                raise ValueError(error_msg)
        
        # 2. Create Daily_Return if it doesn't exist
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # 3. Ensure we have enough data
        if len(df) < 250:
            logging.warning(f"Warning: Not enough data for {ticker} to calculate all features")
            return df
        
        # 4. Price-based features
        windows = [5, 10, 20, 50, 200]
        for w in windows:
            df[f'SMA_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()
        
        # 5. Price crossing signals
        df['SMA_5_10_cross'] = np.where(df['SMA_5'] > df['SMA_10'], 1, -1)
        df['SMA_10_20_cross'] = np.where(df['SMA_10'] > df['SMA_20'], 1, -1)
        df['SMA_50_200_cross'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
        
        # 6. Relative price position
        df['Price_to_SMA_50'] = df['Close'] / df['SMA_50'].replace(0, np.nan)
        df['Price_to_SMA_200'] = df['Close'] / df['SMA_200'].replace(0, np.nan)
        
        # 7. Volatility features
        volatility_windows = [10, 20, 50]
        for w in volatility_windows:
            df[f'Volatility_{w}'] = df['Daily_Return'].rolling(window=w, min_periods=1).std()
        
        # 8. Range and gaps
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Open'].replace(0, np.nan) * 100
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1).replace(0, np.nan) * 100
        
        # 9. Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        
        # 10. MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 11. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 12. Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'].replace(0, np.nan)
        
        # 13. Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
        
        # 14. Return over different periods
        df['Return_5d'] = df['Close'].pct_change(5) * 100
        df['Return_10d'] = df['Close'].pct_change(10) * 100
        df['Return_20d'] = df['Close'].pct_change(20) * 100
        
        # 15. Volume features
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20'].replace(0, np.nan)
        
        # 16. Price/Volume relationship
        df['Price_Volume_Ratio'] = df['Close'] * df['Volume']
        
        # 17. Momentum indicators
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5).replace(0, np.nan) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10).replace(0, np.nan) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20).replace(0, np.nan) - 1
        
        # 18. Date features
        df['Month'] = df.index.month
        df['Day_of_week'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        
        # 19. Cyclic encoding
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Weekday_sin'] = np.sin(2 * np.pi * df['Day_of_week'] / 5)
        df['Weekday_cos'] = np.cos(2 * np.pi * df['Day_of_week'] / 5)
        
        # 20. Target variable
        df['Next_Close'] = df['Close'].shift(-1)
        
        # 21. Final cleaning
        initial_count = len(df)
        df_clean = df.ffill().bfill().dropna()
        removed_count = initial_count - len(df_clean)
        logging.info(f"Removed {removed_count} rows with NaN values ({(removed_count/initial_count)*100:.2f}%)")
        
        return df_clean
    
    def prepare_modeling_data(self, df, ticker, test_size=0.2):
        """
        Prepare data for modeling by splitting into train/test sets and scaling features
        
        Args:
            df: DataFrame with engineered features
            ticker: Stock ticker symbol
            test_size: Proportion of data to use for testing
        
        Returns:
            Dictionary containing datasets and metadata
        """
        logging.info(f"Preparing modeling data for {ticker}")
        
        # Separate features and target
        features = df.drop(['Next_Close'], axis=1)
        target = df['Next_Close']
        
        # Define non-feature columns to exclude
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        # Log feature columns
        logging.info(f"Selected {len(feature_cols)} feature columns for modeling")
        
        # Create X and y
        X = features[feature_cols]
        y = target
        
        # Time-based split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Create dataset dictionary
        datasets = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled_df,
            'X_test_scaled': X_test_scaled_df,
            'feature_names': feature_cols,
            'full_data': df
        }
        
        logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return datasets, scaler
    
    def get_data_transformer_object(self):
        """
        Returns the preprocessor object for transforming features
        
        Returns:
            StandardScaler object
        """
        try:
            # For this stock market prediction task, we'll use a simple StandardScaler
            # This differs from the example as we don't have categorical columns requiring
            # a ColumnTransformer with different pipelines
            
            preprocessor = StandardScaler()
            return preprocessor
            
        except Exception as e:
            logging.error("Error in getting data transformer object")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, raw_data_dir):
        """
        Main method to initiate the data transformation process
        
        Args:
            raw_data_dir: Directory containing raw stock data CSV files
        
        Returns:
            Dictionary with transformation results
        """
        try:
            logging.info("Starting data transformation process")
            
            # Get list of stock data files
            stock_files = [f for f in os.listdir(raw_data_dir) if f.endswith('_raw.csv')]
            tickers = [f.split('_')[0] for f in stock_files]
            
            logging.info(f"Found {len(tickers)} stock data files: {', '.join(tickers)}")
            
            # Dictionary to store all transformed data
            stock_data = {}
            engineered_data = {}
            modeling_data = {}
            scalers = {}
            
            # Process each stock
            for ticker in tickers:
                # Load raw data
                filepath = os.path.join(raw_data_dir, f"{ticker}_raw.csv")
                logging.info(f"Loading raw data for {ticker} from {filepath}")
                
                # Read the CSV file
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                stock_data[ticker] = df
                
                # Calculate daily returns
                stock_data[ticker] = self.calculate_daily_returns(stock_data[ticker])
                
                # Check stationarity
                self.check_stationarity(stock_data[ticker]['Close'], title=f'{ticker} Close Price')
                daily_returns = stock_data[ticker]['Daily_Return'].dropna()
                if not daily_returns.empty:
                    self.check_stationarity(daily_returns, title=f'{ticker} Daily Returns')
                else:
                    logging.info(f"Skipping Daily Returns stationarity check for {ticker} - no valid data")
                
                # Engineer features
                engineered_data[ticker] = self.engineer_features(stock_data[ticker], ticker)
                
                # Save engineered data
                engineered_filepath = os.path.join(
                    self.data_transformation_config.engineered_data_dir, 
                    f"{ticker}_engineered.csv"
                )
                engineered_data[ticker].to_csv(engineered_filepath)
                logging.info(f"Saved engineered data for {ticker} to {engineered_filepath}")
                
                # Prepare data for modeling
                datasets, scaler = self.prepare_modeling_data(engineered_data[ticker], ticker)
                modeling_data[ticker] = datasets
                scalers[ticker] = scaler
                
                # Save modeling datasets
                for name, data in datasets.items():
                    if isinstance(data, list) or name == 'full_data':
                        continue
                    file_path = os.path.join(
                        self.data_transformation_config.train_test_split_dir, 
                        f'{ticker}_{name}.csv'
                    )
                    data.to_csv(file_path)
                    logging.info(f"Saved {name} data for {ticker} to {file_path}")
                
                # Save scaler
                scaler_path = os.path.join(
                    self.data_transformation_config.train_test_split_dir, 
                    f'{ticker}_scaler.pkl'
                )
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logging.info(f"Saved scaler for {ticker} to {scaler_path}")
            
            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=self.get_data_transformer_object()
            )
            logging.info(f"Saved preprocessor object to {self.data_transformation_config.preprocessor_obj_file_path}")
            
            logging.info("Data transformation completed successfully")
            
            return {
                'engineered_data': engineered_data,
                'modeling_data': modeling_data,
                'scalers': scalers,
                'preprocessor_path': self.data_transformation_config.preprocessor_obj_file_path
            }
            
        except Exception as e:
            logging.error("Exception occurred during data transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("Starting data transformation as standalone module")
    
    # Raw data directory
    raw_data_dir = os.path.join('artifacts')
    
    # Initialize data transformation
    data_transformation = DataTransformation()
    
    # Run data transformation
    transformation_results = data_transformation.initiate_data_transformation(raw_data_dir)
    
    logging.info("Data transformation completed successfully when run as standalone module")