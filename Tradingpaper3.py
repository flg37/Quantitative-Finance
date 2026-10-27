import time
import requests
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import lfilter
import yfinance as yf

# Alpaca Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ==========================================
# 0. CONFIGURATION
# ==========================================
API_KEY = ""
SECRET = ""
PAPER = True 

trade_api = TradingClient(API_KEY, SECRET, paper=PAPER)
data_api = StockHistoricalDataClient(API_KEY, SECRET)

# ==========================================
# 1. UNIVERSE GENERATION (S&P 500)
# ==========================================
def get_sp500_tickers():
    """ Scrape S&P 500 tickers from Wikipedia """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers)
        # Wrap in StringIO for pandas 2.0+ compatibility
        df = pd.read_html(io.StringIO(r.text))[0]
        tickers = df['Symbol'].tolist()
        # Clean symbols (Alpaca prefers 'BRK-B' over 'BRK.B')
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Wiki Error: {e}")
        # Robust Fallback
        return ['AAPL','MSFT','GOOG','AMZN','NVDA','TSLA','META','JNJ','V','PG']


def get_data():
    # 1. Build Universe
    tickers = get_sp500_tickers()
    tickers += ['SH', 'SPY']  # SH and SPY exist on Yahoo as well
    tickers = list(set(tickers))

    print(f"Fetching data for {len(tickers)} tickers from yfinance...")

    end = datetime.utcnow()
    start = end - timedelta(days=365)

    # yfinance can fetch in one call
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker"
    )

    # yfinance returns a multi-index (Columns: (field, ticker))
    # Build a Close-price DataFrame with columns = tickers
    close_dict = {}
    for t in tickers:
        try:
            close_dict[t] = data[t]['Close']
        except Exception:
            pass  # skip symbols with no data

    if not close_dict:
        return pd.DataFrame()

    df_close = pd.DataFrame(close_dict).ffill().bfill()
    return df_close

# ==========================================
# 2. MATH HELPERS (Ehlers / RSI)
# ==========================================
def ehlers_super_smoother(data, period=10):
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    b = [c1/2, c1/2]
    a = [1, -c2, -c3]
    return lfilter(b, a, data, axis=0)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. STRATEGY LOGIC ("Pro" Brain)
# ==========================================
def get_weights(df):
    spy = df['SPY']
    # Regime: Bull if Price > 200 SMA
    is_bull = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]
    
    # Filter universe (exclude ETFs)
    stocks = [c for c in df.columns if c not in ['SH', 'SPY']]
    
    # --- SIGNAL GENERATION ---
    log_stocks = np.log(df[stocks])
    
    # 1. Ehlers Momentum
    smooth_fast = pd.DataFrame(ehlers_super_smoother(log_stocks.values, 10), index=df.index, columns=stocks)
    mom_fast = smooth_fast.diff(3).iloc[-1]
    
    smooth_slow = pd.DataFrame(ehlers_super_smoother(log_stocks.values, 30), index=df.index, columns=stocks)
    mom_slow = smooth_slow.diff(10).iloc[-1]
    
    # 2. Volatility
    stock_vol = log_stocks.pct_change().rolling(60).std().iloc[-1]
    
    # 3. Combined Score
    combined_score = (0.6 * mom_fast + 0.4 * mom_slow) / stock_vol
    
    # 4. RSI Overlay
    rsi = calculate_rsi(df[stocks], 14).iloc[-1]
    rsi_adj = (50 - rsi) / 100.0
    final_score = combined_score + rsi_adj
    
    # 5. Junk Filter (Exclude >45% Vol)
    is_junk = stock_vol > 0.45
    final_score[is_junk] = -999.0
    
    # --- WEIGHTING ---
    best_stocks = final_score.nlargest(10).index.tolist()
    weights = {}
    
    if is_bull:
        print(f"Regime: BULL (100% Stocks) - Top Picks: {best_stocks[:3]}")
        # 10% each into Top 10
        for s in best_stocks: weights[s] = 0.10
    else:
        print(f"Regime: BEAR (50% Stocks / 50% SH Hedge)")
        # 5% each into Top 10
        for s in best_stocks: weights[s] = 0.05
        # 50% Hedge (Buy Inverse S&P 500)
        weights['SH'] = 0.50
        
    return weights

# ==========================================
# 4. EXECUTION ENGINE
# ==========================================
def rebalance():
    print(f"\n[{datetime.now()}] Starting Rebalance...")
    
    # 1. Get Data & Weights
    df = get_data()
    if df.empty: 
        print("No data found. Market might be closed or API down.")
        return
        
    tgts = get_weights(df)
    
    # 2. Account Info
    equity = float(trade_api.get_account().equity)
    pos = {p.symbol: float(p.market_value) for p in trade_api.get_all_positions()}
    
    print(f"Account Equity: ${equity:,.2f}")
    
    # 3. SELL Phase (Close old positions)
    for sym in pos:
        if sym not in tgts:
            print(f"Selling {sym} (No longer in target)...")
            trade_api.close_position(sym)
    
    # Wait for sales to process
    time.sleep(3)
    
    # 4. BUY Phase (Open new positions)
    for sym, w in tgts.items():
        target_val = equity * w
        current_val = pos.get(sym, 0)
        diff = target_val - current_val
        
        # Only trade if difference > $10
        if diff > 10:
            try:
                # Use Notional (Fractional Shares)
                req = MarketOrderRequest(
                    symbol=sym,
                    notional=round(diff, 2), # Round to 2 decimals!
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trade_api.submit_order(req)
                print(f"Buy {sym}: ${round(diff, 2)}")
            except Exception as e:
                print(f"Error buying {sym}: {e}")
                
    print("Rebalance Complete.\n")

# ==========================================
# 5. SCHEDULER (Weekly Loop)
# ==========================================
def run_scheduler():
    print("--- Scheduler Started ---")
    print("Bot will run every Monday at 10:00 AM EST.")
    
    while True:
        now = datetime.now()
        # Monday = 0
        if now.weekday() == 0 and now.hour == 15 and now.minute == 0:
            rebalance()
            # Sleep 23 hours to avoid re-triggering same day
            time.sleep(60*60*23)
        else:
            # Check every minute
            time.sleep(60)

if __name__ == "__main__":
    # --- MODE SELECTION ---
    
    # OPTION A: Run once immediately (Testing)
    rebalance()
    
    # OPTION B: Run forever (Production)
    #run_scheduler()