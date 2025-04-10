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

# Setup logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram setup
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
        logger.info(f"Loaded {len(df)} stocks from CSV file")
        return [f"{stock}.NS" for stock in df['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Error loading NSE500 stocks: {str(e)}")
        # Fallback to a small list of stocks
        logger.info("Using fallback stock list")
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

def calculate_mars(stock_symbol, index_symbol, ma_length, ma_type, timeframe='1d', days=100):
    """Calculate MARS indicator and detect crossovers for different timeframes."""
    try:
        # Adjust interval and period based on timeframe
        if timeframe == '1d':
            interval = '1d'
            lookback_days = days
        elif timeframe == '1wk':
            interval = '1wk'
            lookback_days = days * 7  # Need more historical data for weekly
        elif timeframe == '1mo':
            interval = '1mo'
            lookback_days = days * 30  # Need more historical data for monthly
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days+ma_length*2)  # Extra days for MA calculation
        
        logger.info(f"Fetching {timeframe} data for {stock_symbol} from {start_date} to {end_date}")
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval=interval, progress=False)
        index_data = yf.download(index_symbol, start=start_date, end=end_date, interval=interval, progress=False)
        
        if stock_data.empty:
            logger.warning(f"No stock data retrieved for {stock_symbol} on {timeframe}")
            return None
            
        if index_data.empty:
            logger.warning(f"No index data retrieved for {index_symbol} on {timeframe}")
            return None
            
        logger.info(f"Retrieved {len(stock_data)} data points for {stock_symbol} on {timeframe}")
        
        # Calculate MAs
        stock_ma = calculate_ma(stock_data['Close'], ma_length, ma_type)
        index_ma = calculate_ma(index_data['Close'], ma_length, ma_type)
        
        # Calculate percentages
        stock_percent = (stock_data['Close'] - stock_ma) / stock_ma * 100
        index_percent = (index_data['Close'] - index_ma) / index_ma * 100
        
        # Calculate MARS value
        mars_value = stock_percent - index_percent
        
        # Log the latest MARS value
        latest_mars = mars_value.iloc[-1] if not mars_value.empty else None
        logger.info(f"{stock_symbol} {timeframe} latest MARS value: {latest_mars}")
        
        return {
            'stock_data': stock_data,
            'index_data': index_data,
            'mars_value': mars_value,
            'stock_ma': stock_ma,
            'index_ma': index_ma,
            'latest_mars': latest_mars
        }
    except Exception as e:
        logger.error(f"Error calculating MARS for {stock_symbol} on {timeframe}: {str(e)}")
        return None

def generate_chart(stock_symbol, result, timeframe='1d', days_to_show=30, ma_type='EMA', ma_length=50):
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
        
        # Set title based on timeframe
        timeframe_label = "Daily" if timeframe == '1d' else "Weekly" if timeframe == '1wk' else "Monthly"
        ax1.set_title(f'{stock_name} {timeframe_label} Chart')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MARS indicator
        ax2.plot(mars_data.index, mars_data, color='blue', label='MARS')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add horizontal lines at +4 and -3 to show the signal zones
        ax2.axhline(y=4, color='g', linestyle='--', alpha=0.3, label='Buy Zone Limit')
        ax2.axhline(y=-3, color='r', linestyle='--', alpha=0.3, label='Sell Zone Limit')
        
        ax2.set_title(f'MARS Indicator ({timeframe_label})')
        ax2.legend()
        ax2.grid(True)
        
        # Format dates on x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        filename = f'charts/{stock_symbol.replace(".", "_")}_MARS_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename)
        plt.close()
        
        return filename
    except Exception as e:
        logger.error(f"Error generating chart for {stock_symbol} on {timeframe}: {str(e)}")
        return None

def check_crossovers_in_recent_candles(mars_data, num_candles=2):
    """
    Check if there was a crossover in the most recent num_candles.
    Returns a tuple (buy_signal, sell_signal, signal_candle_index)
    """
    if len(mars_data) < num_candles + 1:
        return False, False, None
    
    # Get the most recent candles plus one before to check crossover
    recent_data = mars_data.iloc[-(num_candles+1):]
    
    buy_signal = False
    sell_signal = False
    signal_candle_index = None
    
    # Check each of the last num_candles candles for crossovers
    for i in range(1, num_candles + 1):
        if i >= len(recent_data):
            break
            
        current_value = recent_data.iloc[-i]
        previous_value = recent_data.iloc[-(i+1)]
        
        # Buy signal: previously below 0 and now between 0 and 4
        if previous_value < 0 and 0 <= current_value <= 4:
            buy_signal = True
            signal_candle_index = len(mars_data) - i
            break
            
        # Sell signal: previously above 0 and now between 0 and -3
        elif previous_value > 0 and -3 <= current_value <= 0:
            sell_signal = True
            signal_candle_index = len(mars_data) - i
            break
    
    return buy_signal, sell_signal, signal_candle_index

def check_crossovers():
    """Check for MARS crossovers in all NSE500 stocks across different timeframes."""
    # Configuration
    INDEX = '^NSEI'  # NIFTY 50
    MA_TYPE = 'EMA'  # SMA, EMA, WMA
    MA_LENGTH = 50
    IST = pytz.timezone('Asia/Kolkata')
    
    # Timeframe settings - EXTENDED LOOKBACK PERIODS
    timeframes = {
        '1d': {'label': 'DAILY', 'lookback_days': 14, 'display_days': 30},  # Extended from 7 to 14
        '1wk': {'label': 'WEEKLY', 'lookback_days': 28, 'display_days': 52},  # Extended from 14 to 28
        '1mo': {'label': 'MONTHLY', 'lookback_days': 90, 'display_days': 24}   # Extended from 30 to 90
    }
    
    # Updated emoji definitions that are widely supported
    BUY_EMOJI = "🍏"  # Green apple (universally supported)
    SELL_EMOJI = "🔴"  # Red circle
    
    logger.info(f"Running scan at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load NSE500 stocks
    stock_list = load_nse500_stocks()
    logger.info(f"Processing {len(stock_list)} stocks")
    
    # Track how many alerts we've sent
    alerts_sent = 0
    max_alerts = 30  # Increased limit to accommodate more timeframes
    
    # Track alerts by category for summary
    alert_counts = {
        'DAILY_BUY': 0, 'DAILY_SELL': 0,
        'WEEKLY_BUY': 0, 'WEEKLY_SELL': 0,
        'MONTHLY_BUY': 0, 'MONTHLY_SELL': 0
    }
    
    # Add counters for total stocks processed
    stocks_processed = 0
    stocks_with_data = 0
    
    # Debug counters
    in_buy_zone_count = 0
    in_sell_zone_count = 0
    
    # Send initial status message
    send_telegram_message(f"🔍 *MARS Scan Started*\nProcessing {len(stock_list)} stocks across 3 timeframes...")
    
    # Store stocks for debugging
    stocks_in_zones = {
        'buy_zone': [],
        'sell_zone': []
    }
    
    for stock in stock_list:
        # Stop if we've sent too many alerts
        if alerts_sent >= max_alerts:
            logger.info(f"Reached maximum alerts limit ({max_alerts})")
            break
            
        stocks_processed += 1
        if stocks_processed % 10 == 0:
            logger.info(f"Processed {stocks_processed}/{len(stock_list)} stocks")
            
        for timeframe, settings in timeframes.items():
            try:
                result = calculate_mars(
                    stock, 
                    INDEX, 
                    MA_LENGTH, 
                    MA_TYPE, 
                    timeframe=timeframe, 
                    days=100 if timeframe == '1d' else 200 if timeframe == '1wk' else 365
                )
                
                if result is None:
                    continue
                
                stocks_with_data += 1
                    
                # Check for recent crossovers based on timeframe
                latest_idx = result['mars_value'].index[-1]
                lookback_days = settings['lookback_days']
                
                # For weekly and monthly, we need to look at the index differently
                if timeframe == '1d':
                    lookback_date = latest_idx - timedelta(days=lookback_days)
                elif timeframe == '1wk':
                    lookback_date = latest_idx - timedelta(days=lookback_days)
                else:  # monthly
                    lookback_date = latest_idx - timedelta(days=lookback_days)
                
                # Get data from lookback date to now
                filtered_data = result['mars_value'].loc[lookback_date:]
                if filtered_data.empty:
                    logger.warning(f"No filtered data for {stock} on {timeframe} within lookback period")
                    continue
                
                # Get the latest MARS value
                current_mars = result['latest_mars']
                
                # NEW: Check for crossovers in the last 2 candles
                buy_signal, sell_signal, signal_idx = check_crossovers_in_recent_candles(
                    result['mars_value'], num_candles=2
                )
                
                # Additional debug information
                if 0 <= current_mars <= 4:
                    in_buy_zone_count += 1
                    stocks_in_zones['buy_zone'].append(f"{stock} ({timeframe}): {current_mars:.2f}")
                elif -3 <= current_mars <= 0:
                    in_sell_zone_count += 1
                    stocks_in_zones['sell_zone'].append(f"{stock} ({timeframe}): {current_mars:.2f}")
                
                if buy_signal or sell_signal:
                    logger.info(f"Signal detected for {stock} on {timeframe}: Buy={buy_signal}, Sell={sell_signal}")
                    
                    # Generate chart
                    chart_path = generate_chart(
                        stock, 
                        result, 
                        timeframe=timeframe,
                        days_to_show=settings['display_days'], 
                        ma_type=MA_TYPE, 
                        ma_length=MA_LENGTH
                    )
                    
                    if chart_path is None:
                        continue
                    
                    # Prepare message with emojis
                    stock_name = stock.replace('.NS', '')
                    timeframe_label = settings['label']
                    
                    if buy_signal:
                        crossover_type = "BUY"
                        emoji_prefix = BUY_EMOJI
                        category = f"{timeframe_label}_BUY"
                        alert_counts[category] += 1
                    else:
                        crossover_type = "SELL"
                        emoji_prefix = SELL_EMOJI
                        category = f"{timeframe_label}_SELL"
                        alert_counts[category] += 1
                    
                    message = f"{emoji_prefix} *MARS {timeframe_label} {crossover_type}*: {stock_name}\n\n"
                    message += f"📊 Current MARS Value: {current_mars:.2f}\n"
                    message += f"📅 Date: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}\n"
                    message += f"📈 MA Type: {MA_TYPE}, Length: {MA_LENGTH}\n\n"
                    
                    # Add trend strength indicator
                    if buy_signal:
                        if current_mars > 2:
                            message += "💪 Strong Bullish Signal (In Buy Zone 0 to +4)"
                        else:
                            message += "👍 Bullish Signal (In Buy Zone 0 to +4)"
                    else:
                        if current_mars < -1.5:
                            message += "💪 Strong Bearish Signal (In Sell Zone 0 to -3)"
                        else:
                            message += "👎 Bearish Signal (In Sell Zone 0 to -3)"
                    
                    # Send to Telegram
                    success = send_telegram_message(message, chart_path)
                    
                    if success:
                        logger.info(f"Alert sent for {stock}: MARS {timeframe_label} {crossover_type}")
                        alerts_sent += 1
                    else:
                        logger.warning(f"Failed to send alert for {stock}")
                    
                    # Remove chart file to save space
                    os.remove(chart_path)
                
            except Exception as e:
                logger.error(f"Error processing {stock} on {timeframe}: {str(e)}")
    
    # Debug logging for stocks in zones
    logger.info(f"Total stocks in buy zone (0 to +4): {in_buy_zone_count}")
    logger.info(f"Total stocks in sell zone (0 to -3): {in_sell_zone_count}")
    
    if in_buy_zone_count > 0:
        logger.info(f"Sample stocks in buy zone: {', '.join(stocks_in_zones['buy_zone'][:5])}")
    if in_sell_zone_count > 0:
        logger.info(f"Sample stocks in sell zone: {', '.join(stocks_in_zones['sell_zone'][:5])}")
    
    # Send summary message
    summary = f"🔍 *MARS Scan Complete*\n\n"
    summary += f"Processed {stocks_processed} stocks, {stocks_with_data} had valid data\n\n"
    summary += "📊 *BUY Signals:*\n"
    summary += f"{BUY_EMOJI} DAILY: {alert_counts['DAILY_BUY']}\n"
    summary += f"{BUY_EMOJI} WEEKLY: {alert_counts['WEEKLY_BUY']}\n"
    summary += f"{BUY_EMOJI} MONTHLY: {alert_counts['MONTHLY_BUY']}\n\n"
    summary += "📊 *SELL Signals:*\n"
    summary += f"{SELL_EMOJI} DAILY: {alert_counts['DAILY_SELL']}\n"
    summary += f"{SELL_EMOJI} WEEKLY: {alert_counts['WEEKLY_SELL']}\n"
    summary += f"{SELL_EMOJI} MONTHLY: {alert_counts['MONTHLY_SELL']}\n\n"
    summary += f"Total: {alerts_sent} alerts among {len(stock_list)} stocks"
    
    # Send extended debug information if no alerts were found
    if alerts_sent == 0:
        debug_info = "\n\n📊 *Debug Information:*\n"
        debug_info += f"- Stocks in Buy Zone (0 to +4): {in_buy_zone_count}\n"
        debug_info += f"- Stocks in Sell Zone (0 to -3): {in_sell_zone_count}\n\n"
        
        if in_buy_zone_count > 0 or in_sell_zone_count > 0:
            debug_info += "Some stocks are in signal zones but didn't meet the crossover criteria.\n"
            debug_info += "Consider using a broader detection approach if you want to include these stocks."
        else:
            debug_info += "No stocks found in any signal zones. This could be because:\n"
            debug_info += "1. Market is stable with no significant moves\n"
            debug_info += "2. Lookback period might be too short\n"
            debug_info += "3. There might be data availability issues"
            
        summary += debug_info
    
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
