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
import pandas_ta as ta  # Import pandas_ta

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
            'text': "🔔 *TEST MESSAGE*\nADX Scan system is online",
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

def get_historical_adx_data(stock_symbol, length=14, timeframe='1d', days=100):
    """Fetch historical data and calculate ADX indicators using pandas_ta."""
    try:
        # Adjust interval and period based on timeframe
        if timeframe == '1d':
            interval = '1d'
            lookback_days = days
        elif timeframe == '1wk':
            interval = '1wk'
            lookback_days = days * 7
        elif timeframe == '1mo':
            interval = '1mo'
            lookback_days = days * 30
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + length * 2)

        logger.info(f"Fetching {timeframe} data for {stock_symbol} from {start_date} to {end_date}")
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval=interval, progress=False)

        if stock_data.empty:
            logger.warning(f"No stock data retrieved for {stock_symbol} on {timeframe}")
            return None

        stock_data.ta.adx(length=length, append=True)
        return stock_data[['Open', 'Volume', f'ADX_{length}', f'DMP_{length}', f'DMN_{length}']]

    except Exception as e:
        logger.error(f"Error calculating ADX for {stock_symbol} on {timeframe}: {str(e)}")
        return None

def check_adx_crossovers(adx_data, adx_length=14, num_candles=2):
    """
    Check for ADX DI+ crossing above DI- for buy and vice versa for sell.
    """
    if adx_data is None or len(adx_data) < num_candles + 1:
        return False, False, None

    recent_data = adx_data.iloc[-(num_candles+1):]

    buy_signal = False
    sell_signal = False
    signal_candle_index = None

    for i in range(1, num_candles + 1):
        if i >= len(recent_data):
            break

        current = recent_data.iloc[-i]
        previous = recent_data.iloc[-(i+1)]

        current_di_plus = current[f'DMP_{adx_length}']
        current_di_minus = current[f'DMN_{adx_length}']

        previous_di_plus = previous[f'DMP_{adx_length}']
        previous_di_minus = previous[f'DMN_{adx_length}']

        # Buy Signal: DI+ crosses above DI-
        if previous_di_plus < previous_di_minus and current_di_plus > current_di_minus:
            buy_signal = True
            signal_candle_index = len(adx_data) - i
            break

        # Sell Signal: DI- crosses above DI+
        elif previous_di_plus > previous_di_minus and current_di_plus < current_di_minus:
            sell_signal = True
            signal_candle_index = len(adx_data) - i
            break

    return buy_signal, sell_signal, signal_candle_index

def check_adx_signals():
    """Check for ADX crossovers in all NSE500 stocks across different timeframes."""
    # Configuration
    ADX_LENGTH = 14
    ADX_THRESHOLD = 20
    IST = pytz.timezone('Asia/Kolkata')

    # Timeframe settings
    timeframes = {
        '1d': {'label': 'DAILY', 'lookback_days': 14, 'display_days': 30},
        '1wk': {'label': 'WEEKLY', 'lookback_days': 28, 'display_days': 52},
        '1mo': {'label': 'MONTHLY', 'lookback_days': 90, 'display_days': 24}
    }

    BUY_EMOJI = "🍏"
    SELL_EMOJI = "🔴"

    logger.info(f"Running ADX crossover scan at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")

    stock_list = load_nse500_stocks()
    logger.info(f"Processing {len(stock_list)} stocks")

    alerts_sent = 0
    max_alerts = 30

    alert_counts = {
        'DAILY_BUY': 0, 'DAILY_SELL': 0,
        'WEEKLY_BUY': 0, 'WEEKLY_SELL': 0,
        'MONTHLY_BUY': 0, 'MONTHLY_SELL': 0
    }

    stocks_processed = 0
    stocks_with_data = 0

    send_telegram_message(f"🔍 *ADX Crossover Scan Started*\nProcessing {len(stock_list)} stocks across 3 timeframes...")

    for stock in stock_list:
        if alerts_sent >= max_alerts:
            logger.info(f"Reached maximum alerts limit ({max_alerts})")
            break

        stocks_processed += 1
        if stocks_processed % 10 == 0:
            logger.info(f"Processed {stocks_processed}/{len(stock_list)} stocks")

        for timeframe, settings in timeframes.items():
            try:
                adx_data = get_historical_adx_data(
                    stock,
                    length=ADX_LENGTH,
                    timeframe=timeframe,
                    days=100 if timeframe == '1d' else 200 if timeframe == '1wk' else 365
                )

                if adx_data is None:
                    continue

                stocks_with_data += 1

                buy_signal, sell_signal, signal_idx = check_adx_crossovers(adx_data, adx_length=ADX_LENGTH, num_candles=2)

                if buy_signal or sell_signal:
                    logger.info(f"ADX Crossover detected for {stock} on {timeframe}: Buy={buy_signal}, Sell={sell_signal}")

                    # Generate chart
                    chart_path = generate_adx_chart(
                        stock,
                        adx_data,
                        timeframe=timeframe,
                        days_to_show=settings['display_days'],
                        adx_length=ADX_LENGTH
                    )

                    if chart_path is None:
                        continue

                    stock_name = stock.replace('.NS', '')
                    timeframe_label = settings['label']

                    if buy_signal:
                        signal_type = "BUY"
                        emoji_prefix = BUY_EMOJI
                        category = f"{timeframe_label}_BUY"
                        alert_counts[category] += 1
                    elif sell_signal:
                        signal_type = "SELL"
                        emoji_prefix = SELL_EMOJI
                        category = f"{timeframe_label}_SELL"
                        alert_counts[category] += 1
                    else:
                        continue # Should not happen if the if condition was met

                    message = f"{emoji_prefix} *ADX Crossover ({timeframe_label} {signal_type})*: {stock_name}\n\n"
                    message += f"📊 DI+({ADX_LENGTH}): {adx_data[f'DMP_{ADX_LENGTH}'].iloc[-1]:.2f}\n"
                    message += f"🔴 DI-({ADX_LENGTH}): {adx_data[f'DMN_{ADX_LENGTH}'].iloc[-1]:.2f}\n"
                    if adx_data[f'ADX_{ADX_LENGTH}'].iloc[-1] > ADX_THRESHOLD:
                        message += f"💪 ADX({ADX_LENGTH}) above threshold ({ADX_THRESHOLD:.0f}) indicating strong trend.\n"
                    message += f"📅 Date: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                    success = send_telegram_message(message, chart_path)

                    if success:
                        logger.info(f"Alert sent for {stock}: ADX Crossover {timeframe_label} {signal_type}")
                        alerts_sent += 1
                    else:
                        logger.warning(f"Failed to send alert for {stock}")

                    os.remove(chart_path)

            except Exception as e:
                logger.error(f"Error processing {stock} on {timeframe}: {str(e)}")

    summary = f"🔍 *ADX Crossover Scan Complete*\n\n"
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

    send_telegram_message(summary)

    return alerts_sent

def generate_adx_chart(stock_symbol, adx_data, timeframe='1d', days_to_show=30, adx_length=14):
    """Generate chart with stock price and ADX indicators."""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

        end_idx = len(adx_data)
        start_idx = max(0, end_idx - days_to_show)

        plot_data = adx_data.iloc[start_idx:end_idx].copy()
        plot_data.index = pd.to_datetime(plot_data.index) # Ensure datetime index

        # Plot stock price
        ax1.plot(plot_data.index, plot_data['Open'], label='Open Price')
        ax1.set_title(f'{stock_symbol.replace(".NS", "")} {timeframe.upper()} Chart')
        ax1.legend()
        ax1.grid(True)

        # Plot ADX, DI+, DI-
        ax2.plot(plot_data.index, plot_data[f'ADX_{adx_length}'], color='blue', label=f'ADX({adx_length})')
        ax2.plot(plot_data.index, plot_data[f'DMP_{adx_length}'], color='green', label=f'DI+({adx_length})')
        ax2.plot(plot_data.index, plot_data[f'DMN_{adx_length}'], color='red', label=f'DI-({adx_length})')
        ax2.axhline(y=20, color='k', linestyle='--', alpha=0.5, label='ADX Threshold (20)')
        ax2.set_title(f'ADX and Directional Indicators ({timeframe.upper()})')
        ax2.legend()
        ax2.grid(True)

        # Format dates on x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        filename = f'charts/{stock_symbol.replace(".", "_")}_ADX_CROSSOVER_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename)
        plt.close()

        return filename
    except Exception as e:
        logger.error(f"Error generating ADX chart for {stock_symbol} on {timeframe}: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting ADX crossover signal scan")

    # Test Telegram connection first
    if test_telegram_connection():
        logger.info("Telegram connection successful, proceeding with scan")
        # Run the ADX signal check
        alerts = check_adx_signals()
        logger.info(f"Scan complete. {alerts} alerts sent.")
    else:
        logger.error("Telegram connection failed, please check your token and chat ID")
