import os
import sys
import logging
from datetime import datetime
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer



def main():
    """Main training pipeline execution"""
    try:
        logging.info("Starting training pipeline")
        
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        # Get tickers to train on (you could also load these from a config file)
        tickers = ['AAPL']
        model_trainer.set_tickers(tickers)
        
        # Dictionary to store all results
        all_results = {}
        
        # Train models for each ticker
        for ticker in tickers:
            try:
                logging.info(f"\n{'='*50}")
                logging.info(f"Processing ticker: {ticker}")
                logging.info(f"{'='*50}")
                
                # Train models for this ticker
                ticker_results = model_trainer.initiate_model_training(ticker)
                all_results[ticker] = ticker_results
                
                # Compare models for this ticker
                comparison = model_trainer.compare_all_models(ticker, ticker_results)
                logging.info(f"\nModel comparison for {ticker}:")
                logging.info(comparison)
                logging.info(f"\nCompleted processing for {ticker}")
                
            except Exception as e:
                logging.error(f"Error processing ticker {ticker}: {str(e)}", exc_info=True)
                continue
        
        logging.info("\nTraining pipeline completed successfully")
        
        return all_results
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        results = main()
        
        # Print summary of results for AAPL only
        logging.info("\nTraining Pipeline Summary:")
        for ticker, ticker_results in results.items():
            logging.info(f"\nResults for {ticker}:")
            if 'arima' in ticker_results and 'metrics' in ticker_results['arima']:
                logging.info(f"ARIMA RMSE: {ticker_results['arima']['metrics']['RMSE']:.4f}")
            if 'sarimax' in ticker_results and 'metrics' in ticker_results['sarimax']:
                logging.info(f"SARIMAX RMSE: {ticker_results['sarimax']['metrics']['RMSE']:.4f}")  # Fixed to reference 'sarimax'
            if 'gb' in ticker_results and 'test_metrics' in ticker_results['gb']:
                logging.info(f"Gradient Boosting RMSE: {ticker_results['gb']['test_metrics']['RMSE']:.4f}")
            if 'xgb' in ticker_results and 'test_metrics' in ticker_results['xgb']:
                logging.info(f"XGBoost RMSE: {ticker_results['xgb']['test_metrics']['RMSE']:.4f}")
            if 'lgb' in ticker_results and 'test_metrics' in ticker_results['lgb']:
                logging.info(f"LightGBM RMSE: {ticker_results['lgb']['test_metrics']['RMSE']:.4f}")
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise CustomException(e, sys)