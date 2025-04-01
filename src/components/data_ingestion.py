import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from scipy import stats
from dataclasses import dataclass

# Import custom exception and logging
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # Ensure artifacts directory exists
        os.makedirs(self.ingestion_config.raw_data_path, exist_ok=True)
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Suppressing warnings
            warnings.filterwarnings('ignore')
            
            # Setting plotting styles (for potential future use)
            plt.style.use('fivethirtyeight')
            sns.set_style('whitegrid')
            
            # Define stock tickers and their names
            tickers = ['AAPL', 'GOOGL', 'META', 'MSFT', 'NVDA']
            ticker_names = {
                'AAPL': 'Apple', 
                'GOOGL': 'Google', 
                'META': 'Meta',
                'MSFT': 'Microsoft', 
                'NVDA': 'NVIDIA'
            }
            
            logging.info(f"Defined tickers: {', '.join(tickers)}")
            
            # Define the time period for historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)  # 5 years of data
            
            logging.info(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Fetch data for all stocks
            stock_data = {}
            for ticker in tickers:
                logging.info(f"Downloading data for {ticker_names[ticker]} ({ticker})...")
                stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)
                logging.info(f"Downloaded {len(stock_data[ticker])} rows of data for {ticker}")
                
                # Inspect data
                self._inspect_data(stock_data[ticker], ticker, ticker_names[ticker])
            
            # Clean all datasets
            cleaned_stock_data = {}
            for ticker, df in stock_data.items():
                logging.info(f"Cleaning data for {ticker}...")
                cleaned_stock_data[ticker] = self._clean_stock_data(df, ticker)
                logging.info(f"Cleaning complete for {ticker}")
            
            # Check for outliers in each stock
            for ticker, df in stock_data.items():
                logging.info(f"Detecting outliers for {ticker_names[ticker]} ({ticker})...")
                self._detect_outliers(df, ticker, ticker_names[ticker])
            
            # Save raw data to artifacts folder
            for ticker, df in stock_data.items():
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                filepath = os.path.join(self.ingestion_config.raw_data_path, f"{ticker}_raw.csv")
                df.to_csv(filepath)
                logging.info(f"Saved {ticker} raw data to {filepath}")
            
            logging.info("Data ingestion completed successfully")
            
            return self.ingestion_config.raw_data_path
            
        except Exception as e:
            logging.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)
    
    def _inspect_data(self, df, ticker, ticker_name):
        """Helper method to inspect data and log the results"""
        logging.info(f"--- {ticker_name} ({ticker}) Data Overview ---")
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logging.info(f"First few rows:\n{df.head().to_string()}")
        logging.info(f"Basic statistics:\n{df.describe().to_string()}")
        logging.info(f"Missing values:\n{df.isnull().sum().to_string()}")
    
    def _clean_stock_data(self, df, ticker):
        """Clean the stock data by handling missing values and detecting outliers"""
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Handle missing values - use forward fill for financial time series
        if cleaned_df.isnull().sum().sum() > 0:
            missing_count = cleaned_df.isnull().sum().sum()
            logging.info(f"Filling {missing_count} missing values for {ticker}")
            cleaned_df = cleaned_df.fillna(method='ffill')
            # Use backward fill for any remaining NaNs (e.g., at the beginning)
            cleaned_df = cleaned_df.fillna(method='bfill')
            logging.info(f"Remaining missing values after filling: {cleaned_df.isnull().sum().sum()}")
        
        # Check for outliers using IQR method
        Q1 = cleaned_df.quantile(0.25)
        Q3 = cleaned_df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Flag potential outliers
        outliers = ((cleaned_df < (Q1 - 1.5 * IQR)) | (cleaned_df > (Q3 + 1.5 * IQR)))
        outlier_count = outliers.sum().sum()
        
        if outlier_count > 0:
            logging.info(f"Detected {outlier_count} potential outliers for {ticker}")
            for column in outliers.columns:
                col_outliers = outliers[column].sum()
                if col_outliers > 0:
                    logging.info(f"Column {column}: {col_outliers} outliers")
            # Note: We don't remove outliers as they might be valid market movements
        
        # Ensure the index is datetime and sorted
        cleaned_df.index = pd.to_datetime(cleaned_df.index)
        cleaned_df = cleaned_df.sort_index()
        
        return cleaned_df
    
    def _detect_outliers(self, df, ticker, ticker_name, threshold=3):
        """Detect outliers using z-score method"""
        logging.info(f"--- Outlier Detection for {ticker_name} ({ticker}) using Z-score ---")
        outliers = {}
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in columns:
            z_scores = stats.zscore(df[col])
            outlier_rows = df[np.abs(z_scores) > threshold]
            outliers[col] = outlier_rows
            
            if len(outlier_rows) > 0:
                percent = (len(outlier_rows) / len(df)) * 100
                logging.info(f"{col}: {len(outlier_rows)} outliers detected ({percent:.2f}%)")
                # Log a sample of outliers (up to 3)
                sample = outlier_rows.head(3)
                logging.info(f"Sample outliers for {col}:\n{sample.to_string()}")
            else:
                logging.info(f"{col}: No outliers detected")
        
        return outliers

if __name__ == "__main__":
    logging.info("Starting data ingestion process")
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()
    logging.info(f"Data ingestion completed. Raw data saved to {raw_data_path}")