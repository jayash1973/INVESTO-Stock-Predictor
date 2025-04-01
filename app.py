import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from PIL import Image
import pickle

from src.pipeline.predict_pipeline import PredictPipeline

import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="INVESTO Pro Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional trading interface
st.markdown("""
<style>
    :root {
        --primary:rgb(45, 100, 181);
        --secondary: #4e79a7;
        --accent: #e45756;
        --background: #1e1e1e;
        --card: #333333;
    }
    
    .main {
        background-color: var(--background);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--primary);
        color: white;
    }
    
    .stButton>button {
        background-color: var(--accent);
        color: white;
        border-radius: 4px;
        border: none;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #c04a4a;
        transform: translateY(-1px);
    }
    
    .metric-card {
        background: var(--card);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        border-left: 4px solid var(--accent);
    }
    
    .model-card {
        background: var(--card);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: transform 0.3s;
    }
    
    .model-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .tab-content {
        background: var(--card);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--primary);
    }
    
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    
    .volatility-indicator {
        padding: 8px 12px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .data-table {
        font-size: 0.85rem;
    }

    .viz-container {
        background: var(--card);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
    
    .prediction-container {
        background: linear-gradient(to right,rgb(46, 100, 181),rgb(77, 149, 225));
        color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .indicator-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: 500;
        margin-right: 8px;
    }
    
    .tooltip-custom {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# Load logo and training results
@st.cache_data
# Load logo and training results
@st.cache_data
def load_training_results_and_logo():
    logo_path = 'docs/images/logo.jfif'
    training_results_path = 'artifacts/training_results.pkl'
    
    # Check if the logo file exists
    if not os.path.exists(logo_path):
        # Create a blank image as placeholder
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (200, 100), color=(42, 63, 95))
            d = ImageDraw.Draw(img)
            # Try to add text, but handle if font not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                d.text((40, 40), "INVESTO", fill=(255, 255, 255), font=font)
            except:
                d.text((40, 40), "INVESTO", fill=(255, 255, 255))
            logo = img
        except:
            logo = None
    else:
        try:
            logo = Image.open(logo_path)
        except:
            logo = None
    
    # Load actual training results from the pickle file
    if os.path.exists(training_results_path):
        try:
            with open(training_results_path, 'rb') as f:
                training_results = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load training results: {str(e)}")
            training_results = {}
    else:
        st.warning("Training results file not found.")
        training_results = {}
    
    return logo, training_results

logo, training_results = load_training_results_and_logo()

def compute_rsi(series, window=14):
    """Compute the Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def display_model_comparison(ticker):
    if ticker in training_results:
        model_comparison = training_results[ticker]
        st.markdown(f"### === Model Comparison for {ticker} === ###")
        st.markdown("**Model Performance Comparison:**")
        
        # Create a DataFrame for model performance
        performance_data = []
        for model, metrics in model_comparison.items():
            # Check if the required metrics exist
            if 'RMSE' in metrics and 'MAE' in metrics and 'R2' in metrics:
                performance_data.append({
                    "Model": model,
                    "RMSE": metrics['RMSE'],
                    "Type": "Time Series" if model in ['arima', 'sarimax'] else "Tree-based",
                    "MAE": metrics['MAE'],
                    "MAPE": metrics.get('MAPE', 'N/A'),
                    "R2": metrics['R2'],
                    "Time": metrics.get('time', 'N/A')  # Assuming 'time' might be present
                })
            else:
                st.warning(f"Metrics for model {model} are not available.")
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)

            # Define a function to highlight SARIMAX results
            def highlight_sarimax(row):
                return ['background-color: blue' if row['Model'] == 'sarimax' else '' for _ in row]

            # Apply the highlighting function
            styled_df = performance_df.style.apply(highlight_sarimax, axis=1)

            st.dataframe(styled_df, use_container_width=True)
            
            # Best model
            best_model = performance_df.loc[performance_df['RMSE'].idxmin()]
            st.markdown(f"**Best model:** {best_model['Model']} with RMSE: {best_model['RMSE']:.4f}")
        else:
            st.warning("No valid performance data available.")
    else:
        st.warning(f"No training results found for {ticker}.")

# Enhanced data loading with comprehensive technical indicators
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        
        # Fix MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Basic technical indicators
        # Moving Averages
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['SMA_200'] = data['Close'].rolling(200).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands
        data['20MA'] = data['Close'].rolling(20).mean()
        data['20STD'] = data['Close'].rolling(20).std()
        data['Upper_Band'] = data['20MA'] + (data['20STD'] * 2)
        data['Lower_Band'] = data['20MA'] - (data['20STD'] * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Additional indicators
        # Daily Returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Volatility (rolling standard deviation of returns)
        data['Volatility'] = data['Daily_Return'].rolling(21).std() * np.sqrt(252) * 100
        
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Momentum (14-day)
        data['Momentum_14'] = data['Close'].diff(14) 
        
        # Volume-related indicators
        data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        
        # Volume-related indicators
        data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        
        # Now safe with single-level columns
        if 'Volume_SMA_20' in data:
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
        
        return data.dropna()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Sidebar Configuration
with st.sidebar:
    if logo:
        st.image(logo, width=200)
    st.title("INVESTO Pro Trader")
    st.markdown("""
    **Professional-grade stock analysis and prediction tools**
    """)
    
    # Ticker selection
    available_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    selected_ticker = st.selectbox("Select Stock Ticker", available_tickers)
    
    # Date range selection
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*2)
    
    # Handle date range selection safely
    try:
        date_range = st.date_input(
            "Select Date Range",
            value=(start_date.date(), end_date.date()),
            min_value=(end_date - timedelta(days=365*5)).date(),
            max_value=end_date.date()
        )
        
        # Handle single date selection
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            # If user selects a single date
            start_date = date_range
            end_date = date_range
    except Exception as e:
        st.warning("Date selection error. Using default 2-year range.")
        start_date = (end_date - timedelta(days=365*2)).date()
        end_date = end_date.date()
    
    # Technical indicators
    st.markdown("---")
    st.markdown("**Technical Indicators**")
    indicators = st.multiselect(
        "Select Indicators",
        ['SMA 50/200', 'EMA 20', 'Bollinger Bands', 'RSI', 'MACD', 'Volatility', 'ATR'],
        default=['SMA 50/200', 'RSI', 'Bollinger Bands']
    )
    
    # Visualization options
    st.markdown("---")
    st.markdown("**Visualization Options**")
    chart_type = st.radio(
        "Primary Chart Type",
        ["Candlestick", "Line", "OHLC"]
    )
    
    show_volume = st.checkbox("Show Volume", value=True)
    show_returns_dist = st.checkbox("Show Returns Distribution", value=True)
    
    # Analysis timeframe
    st.markdown("---")
    st.markdown("**Analysis Timeframe**")
    forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=5)
    
    st.markdown("---")
    st.markdown("""
    **Model Settings**
    - Uses ARIMA, SARIMAX, and ensemble models
    - Trained on OHLCV + technical indicators
    - 30-day lookback for predictions
    """)

# Main content
st.title("INVESTO APPLE Professional Trading Analysis")
st.markdown("---")

# Load data with error handling
stock_data = load_stock_data(selected_ticker, start_date, end_date)
if stock_data is None:
    st.error("Failed to load stock data. Please try different parameters.")
    st.stop()


# Overview metrics
overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

# Get current price, change, and other key metrics
current_price = stock_data['Close'].iloc[-1]
prev_price = stock_data['Close'].iloc[-2]
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

with overview_col1:
    st.metric(
        label="Current Price",
        value=f"${current_price:.2f}",
        delta=f"{price_change_pct:.2f}%"
    )

with overview_col2:
    high_52w = stock_data['High'].tail(252).max()
    low_52w = stock_data['Low'].tail(252).min()
    pct_from_high = ((current_price - high_52w) / high_52w) * 100
    
    st.metric(
        label="52-Week Range",
        value=f"${low_52w:.2f} - ${high_52w:.2f}",
        delta=f"{pct_from_high:.2f}% from high"
    )

with overview_col3:
    avg_volume = stock_data['Volume'].mean()
    last_volume = stock_data['Volume'].iloc[-1]
    volume_change = ((last_volume - avg_volume) / avg_volume) * 100
    
    st.metric(
        label="Volume",
        value=f"{last_volume:,.0f}",
        delta=f"{volume_change:.2f}% vs avg"
    )

with overview_col4:
    if 'Volatility' in stock_data.columns:
        current_vol = stock_data['Volatility'].iloc[-1]
        avg_vol = stock_data['Volatility'].mean()
        vol_change = current_vol - avg_vol
        
        st.metric(
            label="Volatility (Ann.)",
            value=f"{current_vol:.2f}%",
            delta=f"{vol_change:.2f}%",
            delta_color="inverse"  # Higher volatility is usually negative
        )
    else:
        st.metric(
            label="Volatility (Ann.)",
            value="N/A"
        )

# Main visualization tabs
viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
    "Price & Technical Analysis", 
    "Volume & Returns", 
    "Correlation & Volatility", 
    "Advanced Indicators"
])

with viz_tab1:
    # Main chart with indicators
    st.subheader("Price Chart & Technical Indicators")
    
    # Create figure with subplots for price and indicators
    rows = 2 if 'RSI' in indicators or 'MACD' in indicators else 1
    subplot_row_heights = [0.7, 0.3] if rows == 2 else [1]
    
    fig = make_subplots(
        rows=rows, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05, 
        row_heights=subplot_row_heights,
        specs=[[{"secondary_y": False}], 
               [{"secondary_y": True}]] if rows == 2 else [[{"secondary_y": False}]]
    )
    
    # Price chart
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name="OHLC",
                increasing_line_color='#2ecc71',
                decreasing_line_color='#e74c3c'
            ), row=1, col=1  # Correctly specify row and column here
        )
    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name="OHLC",
                increasing_line_color='#2ecc71',
                decreasing_line_color='#e74c3c'
            ), row=1, col=1  # Correctly specify row and column here
        )
    else:  # Line chart
        fig.add_trace(
            go.Scatter(
                x=stock_data.index, 
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#2c3e50', width=2)
            ), row=1, col=1  # Correctly specify row and column here
        )
    
    # Add selected indicators
    if 'SMA 50/200' in indicators:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index, 
                y=stock_data['SMA_50'], 
                name='SMA 50', 
                line=dict(color='#3498db', width=1.5)
            ), row=1, col=1  # Correctly specify row and column here
        )
        fig.add_trace(
            go.Scatter(
                x=stock_data.index, 
                y=stock_data['SMA_200'], 
                name='SMA 200', 
                line=dict(color='#9b59b6', width=1.5)
            ), row=1, col=1  # Correctly specify row and column here
        )
    
    if 'EMA 20' in indicators:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index, 
                y=stock_data['EMA_20'], 
                name='EMA 20', 
                line=dict(color='#e67e22', width=1.5)
            ), row=1, col=1  # Correctly specify row and column here
        )
    
    if 'Bollinger Bands' in indicators:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index, 
                y=stock_data['Upper_Band'], 
                name='Upper Band', 
                line=dict(color='#95a5a6', width=1)
            ), row=1, col=1  # Correctly specify row and column here
        )
        fig.add_trace(
            go.Scatter(
                x=stock_data.index, 
                y=stock_data['Lower_Band'], 
                name='Lower Band', 
                line=dict(color='#95a5a6', width=1), 
                fill='tonexty'
            ), row=1, col=1  # Correctly specify row and column here
        )
    
    # Bottom indicators (RSI, MACD)
    if rows == 2:
        if 'RSI' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index, 
                    y=stock_data['RSI'], 
                    name='RSI', 
                    line=dict(color='#f39c12', width=1.5)
                ), row=2, col=1  # Correctly specify row and column here
            )
            fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")
            fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
            
            # Add overbought/oversold annotations
            fig.add_annotation(
                x=stock_data.index[-1], 
                y=70,
                text="Overbought",
                showarrow=False,
                xshift=50,
                font=dict(size=10, color="red"),
                row=2, col=1  # Correctly specify row and column here
            )
            fig.add_annotation(
                x=stock_data.index[-1], 
                y=30,
                text="Oversold",
                showarrow=False,
                xshift=50,
                font=dict(size=10, color="green"),
                row=2, col=1  # Correctly specify row and column here
            )
        
        if 'MACD' in indicators:
            fig.add_trace(
                go.Bar(
                    x=stock_data.index, 
                    y=stock_data['MACD'] - stock_data['Signal_Line'],
                    name='MACD Histogram',
                    marker_color=np.where(
                        (stock_data['MACD'] - stock_data['Signal_Line']) > 0,
                        '#2ecc71', '#e74c3c'
                    )
                ), row=2, col=1, secondary_y=True  # Correctly specify row and column here
            )
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index, 
                    y=stock_data['MACD'], 
                    name='MACD', 
                    line=dict(color='#3498db', width=1.5)
                ), row=2, col=1, secondary_y=True  # Correctly specify row and column here
            )
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index, 
                    y=stock_data['Signal_Line'], 
                    name='Signal Line', 
                    line=dict(color='#e67e22', width=1.5)
                ), row=2, col=1, secondary_y=True  # Correctly specify row and column here
            )
    
    # Update layout
    fig.update_layout(
        height=600, 
        title=f"{selected_ticker} Technical Analysis",
        showlegend=True, 
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving Average Crossover Analysis
    if 'SMA 50/200' in indicators:
        st.subheader("Moving Average Crossover Analysis")
        # Calculate if golden cross or death cross is present
        ma_50 = stock_data['SMA_50'].iloc[-1]
        ma_200 = stock_data['SMA_200'].iloc[-1]
        ma_50_prev = stock_data['SMA_50'].iloc[-2]
        ma_200_prev = stock_data['SMA_200'].iloc[-2]
        
        golden_cross = ma_50_prev < ma_200_prev and ma_50 > ma_200
        death_cross = ma_50_prev > ma_200_prev and ma_50 < ma_200
        
        ma_cols = st.columns(3)
        with ma_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SMA 50 vs SMA 200</div>
                <div class="metric-value">{((ma_50 / ma_200) - 1) * 100:.2f}%</div>
                <div>50 SMA: ${ma_50:.2f} | 200 SMA: ${ma_200:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ma_cols[1]:
            cross_status = "Golden Cross" if golden_cross else "Death Cross" if death_cross else "No Recent Crossover"
            cross_color = "#2ecc71" if golden_cross else "#e74c3c" if death_cross else "#7f8c8d"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {cross_color};">
                <div class="metric-label">Moving Average Signal</div>
                <div class="metric-value" style="color: {cross_color};">{cross_status}</div>
                <div>Indicates {'bullish' if golden_cross else 'bearish' if death_cross else 'neutral'} trend</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ma_cols[2]:
            # Simple trend strength indicator
            trend_diff = ((ma_50 / ma_200) - 1) * 100
            trend_strength = "Strong" if abs(trend_diff) > 5 else "Moderate" if abs(trend_diff) > 2 else "Weak"
            trend_type = "Bullish" if trend_diff > 0 else "Bearish"
            trend_color = "#2ecc71" if trend_diff > 0 else "#e74c3c"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {trend_color};">
                <div class="metric-label">Trend Strength</div>
                <div class="metric-value" style="color: {trend_color};">{trend_strength} {trend_type}</div>
                <div>Based on MA separation</div>
            </div>
            """, unsafe_allow_html=True)

with viz_tab2:
    # Volume Analysis
    vol_col1, vol_col2 = st.columns(2)
    
    with vol_col1:
        st.subheader("Volume Analysis")
        
        # Volume chart with moving average
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                marker_color='rgba(126, 142, 158, 0.6)'
            )
        )
        
        if 'Volume_SMA_20' in stock_data.columns:
            fig_vol.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Volume_SMA_20'],
                    name="20-day Volume MA",
                    line=dict(color='#e74c3c', width=1.5)
                )
            )
        
        fig_vol.update_layout(
            height=400,
            title="Volume with 20-day Moving Average",
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Volume",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with vol_col2:  # Ensure this line is correctly indented
        st.subheader("Daily Returns Distribution")
        
        # Daily returns histogram
        returns = stock_data['Daily_Return'].dropna() * 100  # Convert to percentage
        
        fig_returns = go.Figure()
        fig_returns.add_trace(
            go.Histogram(
                x=returns,
                name="Daily Returns",
                marker_color='rgba(78, 121, 167, 0.6)',
                nbinsx=30
            )
        )
        
        # Add normal distribution curve for comparison
        mean_return = returns.mean()
        std_return = returns.std()
        x_range = np.linspace(returns.min(), returns.max(), 100)
        y_norm = 1/(std_return * np.sqrt(2*np.pi)) * np.exp(-(x_range-mean_return)**2 / (2*std_return**2))
        y_norm = y_norm * (len(returns) * (returns.max() - returns.min()) / 30)
        
        fig_returns.add_trace(
            go.Scatter(
                x=x_range,
                y=y_norm,
                name="Normal Distribution",
                line=dict(color='#e74c3c', width=2)
            )
        )
        
        fig_returns.add_vline(
            x=mean_return, 
            line_dash="dash", 
            line_color="#2c3e50",
            annotation_text=f"Mean: {mean_return:.2f}%"
        )
        
        fig_returns.update_layout(
            height=400,
            title=f"Distribution of Daily Returns (Mean: {mean_return:.2f}%, Std Dev: {std_return:.2f}%)",
            template="plotly_white",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            bargap=0.1,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
                
        # Returns statistics
        ret_col1, ret_col2 = st.columns(2)
        with ret_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Positive Days</div>
                <div class="metric-value">{(returns > 0).mean():.1%}</div>
            </div>
            """, unsafe_allow_html=True)
                    
        with ret_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Negative Days</div>
                <div class="metric-value">{(returns < 0).mean():.1%}</div>
            </div>
            """, unsafe_allow_html=True)

with viz_tab3:
    # Correlation and Volatility Analysis
    st.subheader("Feature Correlation Matrix")
    
    # Select numeric columns for correlation
    numeric_cols = stock_data.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = stock_data[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        aspect="auto",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    
    fig_corr.update_layout(
        height=600,
        title="Feature Correlation Heatmap",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Volatility Analysis
    st.subheader("Price Volatility Analysis")
    
    if 'Volatility' in stock_data.columns:
        fig_volatility = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Price with volatility bands
        fig_volatility.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name='Close Price',
                line=dict(color='#2c3e50', width=2)
            ), row=1, col=1
        )
        
        # Add 1 standard deviation bands
        rolling_mean = stock_data['Close'].rolling(21).mean()
        rolling_std = stock_data['Close'].rolling(21).std()
        
        fig_volatility.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=rolling_mean + rolling_std,
                name='Upper Band (1σ)',
                line=dict(color='#95a5a6', width=1)
                # row=1, col=1
        ))
        
        fig_volatility.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=rolling_mean - rolling_std,
                name='Lower Band (1σ)',
                line=dict(color='#95a5a6', width=1),
                fill='tonexty'
                # row=1, col=1
        ))
        
        # Volatility chart
        fig_volatility.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Volatility'],
                name='Annualized Volatility',
                line=dict(color='#e74c3c', width=2)
            ), row=2, col=1
        )
        
        fig_volatility.update_layout(
            height=600,
            title="Price with Volatility Bands and Volatility Trend",
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig_volatility, use_container_width=True)
    else:
        st.warning("Volatility data not available")

with viz_tab4:
    st.subheader("Advanced Technical Indicators")

    # Allow users to select which indicators to display
    indicators_to_display = st.multiselect(
        "Select Indicators to Display",
        ['ATR', 'Momentum', 'Bollinger Bands', 'MACD', 'RSI'],
        default=['ATR', 'Momentum', 'Bollinger Bands', 'MACD', 'RSI']
    )

    # Create a figure for advanced indicators
    fig_advanced = make_subplots(rows=len(indicators_to_display), cols=1, shared_xaxes=True)

    # Plot ATR if selected
    if 'ATR' in indicators_to_display:
        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['ATR'],
                name='Average True Range (ATR)',
                line=dict(color='#3498db', width=2)
            ), row=1, col=1
        )

    # Plot Momentum if selected
    if 'Momentum' in indicators_to_display:
        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Momentum_14'],
                name='14-Day Momentum',
                line=dict(color='#9b59b6', width=2)
            ), row=2, col=1
        )
        fig_advanced.add_hline(
            y=0, 
            line_dash="dot",
            line_color="#7f8c8d",
            row=2, col=1
        )

    # Plot Bollinger Bands if selected
    if 'Bollinger Bands' in indicators_to_display:
        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Upper_Band'],
                name='Upper Band',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=1)
            ), row=3, col=1
        )
        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Lower_Band'],
                name='Lower Band',
                line=dict(color='rgba(0, 255, 0, 0.5)', width=1)
            ), row=3, col=1
        )
        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name='Close Price',
                line=dict(color='#2c3e50', width=2)
            ), row=3, col=1
        )

    # Plot MACD if selected
    if 'MACD' in indicators_to_display:
        exp12 = stock_data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal_line = macd.ewm(span=9, adjust=False).mean()

        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=macd,
                name='MACD',
                line=dict(color='blue', width=2)
            ), row=4, col=1
        )
        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=signal_line,
                name='Signal Line',
                line=dict(color='orange', width=2)
            ), row=4, col=1
        )

    # Plot RSI if selected
    if 'RSI' in indicators_to_display:
        rsi = compute_rsi(stock_data['Close'], window=14)  # Assuming you have a function to compute RSI
        fig_advanced.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=rsi,
                name='RSI',
                line=dict(color='purple', width=2)
            ), row=5, col=1
        )
        fig_advanced.add_hline(y=70, line_dash="dot", line_color="red", row=5, col=1)
        fig_advanced.add_hline(y=30, line_dash="dot", line_color="green", row=5, col=1)

    # Update layout for the advanced indicators plot
    fig_advanced.update_layout(
        height=800,
        title="Advanced Technical Indicators",
        template="plotly_white",
        showlegend=True
    )

    st.plotly_chart(fig_advanced, use_container_width=True)

# Prediction Section
st.header("AI-Powered Price Forecasts")
st.markdown("""
<div class="prediction-container">
    <h3 style="color: white;">Model Consensus Forecast</h3>
    <p>Our ensemble of quantitative models predicts the following price movements:</p>
    <p>The current best model is SARIMX with an overall avg error of 3$</>
</div>
""", unsafe_allow_html=True)


if st.button("Generate Predictions", key="predict_button"):
    if selected_ticker != 'AAPL':
        st.error("Predictions are currently only available for Apple Inc. (AAPL). Please select AAPL from the sidebar.")
    else:
        with st.spinner("Running quantitative models..."):
            try:
                # Initialize the PredictPipeline with the selected ticker and forecast days
                predictor = PredictPipeline(ticker=selected_ticker, forecast_days=forecast_days)
                predictions = predictor.predict()  # Get real predictions
                
                # Get today's date and calculate the prediction date
                today = datetime.today()
                prediction_date = today + timedelta(days=forecast_days)
                formatted_prediction_date = prediction_date.strftime("%Y-%m-%d")  # Format the date
                
                # Display predictions
                cols = st.columns(5)
                model_colors = {'ARIMA': '#3498db', 'SARIMAX': '#9b59b6',
                                'GB': '#2ecc71', 'XGB': '#e74c3c', 'LGB': '#f39c12'}
                
                current_price = stock_data['Close'].iloc[-1]
                prediction_values = []
                
                for idx, (model, data) in enumerate(predictions.items()):
                    with cols[idx]:
                        try:
                            change = ((data['prediction'] / current_price) - 1) * 100
                            st.markdown(f"""
                            <div class="model-card">
                                <h4 style="color: {model_colors[model]};">{model}</h4>
                                <h3>${data['prediction']:,.2f}</h3>
                                <p style="color: {'#2ecc71' if change >=0 else '#e74c3e'};">{'▲' if change >=0 else '▼'} {abs(change):.2f}%</p>
                                <small>Prediction Date: {formatted_prediction_date}</small>  <!-- Updated date display -->
                            </div>
                            """, unsafe_allow_html=True)
                            prediction_values.append(data['prediction'])
                        except Exception as e:
                            st.markdown(f"""
                            <div class="model-card">
                                <h4 style="color: {model_colors[model]};">{model}</h4>
                                <p>Prediction failed: {str(e)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Consensus analysis
                if prediction_values:
                    st.subheader("Model Consensus Analysis")
                    avg_prediction = np.mean(prediction_values)
                    consensus_change = ((avg_prediction / current_price) - 1) * 100
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Consensus Price</div>
                            <div class="metric-value">${avg_prediction:,.2f}</div>
                            <div>Current: ${current_price:,.2f}</div>
                            <div style="color: {'#2ecc71' if consensus_change >=0 else '#e74c3e'}; margin-top: 8px;">
                                {'▲' if consensus_change >=0 else '▼'} {abs(consensus_change):.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        fig_consensus = go.Figure()
                        for model, data in predictions.items():
                            try:
                                fig_consensus.add_trace(go.Scatter(
                                    x=[model],
                                    y=[data['prediction']],
                                    marker_color=model_colors[model],
                                    name=model,
                                    showlegend=False,
                                    marker_size=15
                                ))
                            except:
                                continue
                        
                        fig_consensus.add_hline(
                            y=current_price, 
                            line_dash="dot",
                            annotation_text="Current Price", 
                            line_color="#2c3e50"
                        )
                        
                        fig_consensus.update_layout(
                            title="Model Predictions Comparison",
                            yaxis_title="Price ($)",
                            height=300,
                            template="plotly_white",
                            showlegend=False,
                            xaxis_title="Model"
                        )
                        
                        st.plotly_chart(fig_consensus, use_container_width=True)
                    display_model_comparison('AAPL')
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Data Exploration Section
st.header("Fundamental Data Analysis")
with st.expander("View Historical Data"):
    st.dataframe(stock_data.tail(10), use_container_width=True)

with st.expander("Statistical Summary"):
    st.dataframe(stock_data.describe(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**INVESTO Quantitative Research**  
*Professional Trading Analytics Platform*  
[GitHub Repository](https://github.com/jayash1973/INVESTO-Stock-Predictor) | [Contact Team](mailto:jayashbhardwaj294@gmail.com)
""")