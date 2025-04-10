import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import telegram
import pytz
from datetime import datetime, timedelta
import os
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram setup - the key issue is likely here
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Create directory for charts
if not os.path.exists('charts'):
    os.makedirs('charts')

def test_telegram_connection():
    """Test the Telegram connection by sending a message."""
    try:
        logger.info(f"Testing Telegram connection with chat_id: {TELEGRAM_CHAT_ID}")
        
        # Using direct API call which is more reliable than the python-telegram-bot library
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': "🔔 *TEST MESSAGE*\nMARS Scan system is online",
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=payload)
        result = response.json()
        
        if result.get('ok'):
            logger.info("Test message sent successfully via direct API")
            return True
        else:
            error_msg = result.get('description', 'Unknown error')
            logger.error(f"Test message failed: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"Telegram test failed: {str(e)}")
        return False

def send_telegram_message(message, photo_path=None):
    """Send a message or photo to Telegram using direct API calls."""
    try:
        logger.info(f"Sending Telegram message to chat_id: {TELEGRAM_CHAT_ID}")
        
        if photo_path:
            # Send photo with caption
            url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                payload = {
                    'chat_id': TELEGRAM_CHAT_ID,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, data=payload, files=files)
        else:
            # Send text message
            url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, data=payload)
        
        result = response.json()
        if result.get('ok'):
            logger.info("Message sent successfully")
            return True
        else:
            error_msg = result.get('description', 'Unknown error')
            logger.error(f"Failed to send message: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")
        return False

def load_nse500_stocks():
    """Load NSE500 stocks from CSV file."""
    try:
        df = pd.read_csv('nse500_stocks.csv')
        return [f"{stock}.NS" for stock in df['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Error loading NSE500 stocks: {str(e)}")
        # Fallback to a small list of stocks
        return ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']

def calculate_ma(data, length, ma_type='EMA'):
    """Calculate Moving Average based on type."""
    if ma_type == 'SMA':
        return data.rolling(window=length).mean()
    elif ma_type == 'EMA':
        return data.ewm(span=length, adjust=False).mean()
    elif ma_type == 'WMA':
        weights = np.arange(1, length + 1)
        return data.rolling(window=length).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)

def calculate_mars(stock_symbol, index_symbol, ma_length, ma_type, days=100):
    """Calculate MARS indicator and detect crossovers."""
    try:
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+ma_length)  # Extra days for MA calculation
        
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        index_data = yf.download(index_symbol, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty or index_data.empty:
            return None
        
        # Calculate MAs
        stock_ma = calculate_ma(stock_data['Close'], ma_length, ma_type)
        index_ma = calculate_ma(index_data['Close'], ma_length, ma_type)
        
        # Calculate percentages
        stock_percent = (stock_data['Close'] - stock_ma) / stock_ma * 100
        index_percent = (index_data['Close'] - index_ma) / index_ma * 100
        
        # Calculate MARS value
        mars_value = stock_percent - index_percent
        
        # Detect crossovers
        previous_mars = mars_value.shift(1)
        crossover_up = (previous_mars < 0) & (mars_value > 0)
        crossover_down = (previous_mars > 0) & (mars_value < 0)
        
        return {
            'stock_data': stock_data,
            'index_data': index_data,
            'mars_value': mars_value,
            'crossover_up': crossover_up,
            'crossover_down': crossover_down,
            'stock_ma': stock_ma,
            'index_ma': index_ma,
            'latest_mars': mars_value.iloc[-1] if not mars_value.empty else None
        }
    except Exception as e:
        logger.error(f"Error calculating MARS for {stock_symbol}: {str(e)}")
        return None

def generate_chart(stock_symbol, result, days_to_show=30, ma_type='EMA', ma_length=50):
    """Generate chart with stock price and MARS indicator."""
    try:
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # Get subset of data to display
        end_idx = len(result['stock_data'])
        start_idx = max(0, end_idx - days_to_show)
        
        stock_data = result['stock_data'].iloc[start_idx:end_idx]
        mars_data = result['mars_value'].iloc[start_idx:end_idx]
        
        # Plot stock price
        ax1.plot(stock_data.index, stock_data['Close'], label='Close Price')
        ax1.plot(stock_data.index, result['stock_ma'].iloc[start_idx:end_idx], label=f'{ma_type} {ma_length}')
        stock_name = stock_symbol.replace('.NS', '')
        ax1.set_title(f'{stock_name} Stock Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MARS indicator
        ax2.plot(mars_data.index, mars_data, color='blue', label='MARS')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.set_title('MARS Indicator')
        ax2.legend()
        ax2.grid(True)
        
        # Format dates on x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        filename = f'charts/{stock_symbol.replace(".", "_")}_MARS_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename)
        plt.close()
        
        return filename
    except Exception as e:
        logger.error(f"Error generating chart for {stock_symbol}: {str(e)}")
        return None

def check_crossovers():
    """Check for MARS crossovers in all NSE500 stocks."""
    # Configuration
    INDEX = '^NSEI'  # NIFTY 50
    MA_TYPE = 'EMA'  # SMA, EMA, WMA
    MA_LENGTH = 50
    IST = pytz.timezone('Asia/Kolkata')
    
    print(f"Running scan at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load NSE500 stocks
    stock_list = load_nse500_stocks()
    
    # Track how many alerts we've sent
    alerts_sent = 0
    max_alerts = 10  # Limit number of alerts to avoid flooding
    
    for stock in stock_list:
        # Stop if we've sent too many alerts
        if alerts_sent >= max_alerts:
            break
            
        try:
            result = calculate_mars(stock, INDEX, MA_LENGTH, MA_TYPE)
            
            if result is None:
                continue
                
            # Check for recent crossovers (last 2 days to catch weekend crossovers)
            latest_idx = result['mars_value'].index[-1]
            two_days_ago = latest_idx - timedelta(days=2)
            
            recent_crossover_up = result['crossover_up'].loc[two_days_ago:].any()
            recent_crossover_down = result['crossover_down'].loc[two_days_ago:].any()
            
            if recent_crossover_up or recent_crossover_down:
                # Generate chart
                chart_path = generate_chart(stock, result, ma_type=MA_TYPE, ma_length=MA_LENGTH)
                
                if chart_path is None:
                    continue
                
                # Prepare message with emojis
                stock_name = stock.replace('.NS', '')
                crossover_type = "UP 🚀" if recent_crossover_up else "DOWN 📉"
                current_mars = result['latest_mars']
                
                message = f"🔔 *MARS CROSSOVER {crossover_type}*: {stock_name}\n\n"
                message += f"📊 Current MARS Value: {current_mars:.2f}\n"
                message += f"📅 Date: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}\n"
                message += f"📈 MA Type: {MA_TYPE}, Length: {MA_LENGTH}\n\n"
                
                # Add trend strength indicator
                if recent_crossover_up:
                    if current_mars > 5:
                        message += "💪 Strong Bullish Signal"
                    else:
                        message += "👍 Bullish Signal"
                else:
                    if current_mars < -5:
                        message += "💪 Strong Bearish Signal"
                    else:
                        message += "👎 Bearish Signal"
                
                # Send to Telegram
                success = send_telegram_message(message, chart_path)
                
                if success:
                    logger.info(f"Alert sent for {stock}: MARS CROSSOVER {crossover_type}")
                    alerts_sent += 1
                else:
                    logger.warning(f"Failed to send alert for {stock}")
                
                # Remove chart file to save space
                os.remove(chart_path)
            
        except Exception as e:
            logger.error(f"Error processing {stock}: {str(e)}")
    
    # Send summary message
    summary = f"🔍 *MARS Scan Complete*\n📊 {alerts_sent} crossovers detected among {len(stock_list)} stocks"
    send_telegram_message(summary)
    
    return alerts_sent

if __name__ == "__main__":
    logger.info("Starting MARS crossover scan")
    
    # Test Telegram connection first
    if test_telegram_connection():
        logger.info("Telegram connection successful, proceeding with scan")
        # Run the crossover check
        alerts = check_crossovers()
        logger.info(f"Scan complete. {alerts} alerts sent.")
    else:
        logger.error("Telegram connection failed, please check your token and chat ID")
