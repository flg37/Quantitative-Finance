import requests
from alpaca.trading.requests import LimitOrderRequest
import pandas as pd
import numpy as np
from scipy.stats import linregress
import warnings
import os
import pickle
import time
import json
from datetime import datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import io

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_KEY = ""
SECRET_KEY = ""
PAPER = True  # Set to False for real money (be careful)

def get_sp500_tickers():
    """
    Scrape S&P 500 tickers from Wikipedia using a User-Agent header
    to avoid HTTP 403 Forbidden errors.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                'TSLA', 'META', 'BRK-B', 'V', 'JNJ']

# Get current S&P 500 list
TICKERS = get_sp500_tickers()
# Add safety ETFs
TICKERS += ['SHY', 'GLD', 'TLT']

CONFIG = {
    'start_date': '2020-01-01',
    'rls_memory': 0.98,
    'rebalance_freq': 15,
    'top_n': 10,
    'cache_file': 'live_data_cache.pkl',
    'state_file': 'strategy_state.pkl'
}

# Initialize Alpaca Clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ==========================================
# 2. RLS FILTER & FEATURE LOGIC
# ==========================================
class RLSFilter:
    def __init__(self, n=5, lam=0.98):
        self.n = n
        self.lam = lam
        self.theta = np.zeros(n, dtype=float)
        self.P = np.eye(n, dtype=float) * 10.0

    def adapt(self, x, y):
        if not np.isfinite(x).all() or not np.isfinite(y):
            return
        try:
            y_hat = float(np.dot(x, self.theta))
            error = y - y_hat
            Px = np.dot(self.P, x)
            gain_denom = self.lam + float(np.dot(x, Px)) + 1e-8
            k = Px / gain_denom
            self.theta += k * error
            self.P = (self.P - np.outer(k, Px)) / self.lam
            self.theta = np.clip(self.theta, -3, 3)
            self.P = np.clip(self.P, -100, 100)
        except Exception as e:
            print(f"⚠️ RLS adaptation failed: {e}")
            self.theta = np.zeros(self.n, dtype=float)
            self.P = np.eye(self.n, dtype=float) * 10.0

    def predict(self, x):
        if not np.isfinite(x).all():
            return 0.0
        return float(np.dot(x, self.theta))

def get_features(series, benchmark_series):
    if len(series) < 60:
        return None
    try:
        series = series.iloc[-60:].copy()
        benchmark_series = benchmark_series.iloc[-60:].copy()

        if (series <= 0).any() or (benchmark_series <= 0).any():
            return None

        f_mom = series.iloc[-1] / series.iloc[-21] - 1

        rets = series.pct_change().fillna(0)
        f_vol = rets.tail(20).std()

        y_vals = np.log(series.iloc[-30:].values)
        x_vals = np.arange(len(y_vals))
        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        f_consistency = r_value ** 2

        asset_ret = series.iloc[-1] / series.iloc[-21] - 1
        spy_ret = benchmark_series.iloc[-1] / benchmark_series.iloc[-21] - 1
        f_alpha = asset_ret - spy_ret

        feats = np.array([1.0, f_mom, f_vol, f_consistency, f_alpha], dtype=float)
        if not np.isfinite(feats).all():
            return None
        return feats
    except Exception as e:
        print(f"⚠️ Feature extraction failed: {e}")
        return None

# ==========================================
# 3. ALPACA DATA RETRIEVAL
# ==========================================
def get_data_from_alpaca():
    """
    Download price data from Alpaca.
    Aligns data by date so short-history stocks don't ruin long-history ones.
    """
    print("Downloading data from Alpaca...")
    
    # 5 years to be safe
    end_date = datetime.now().date()
    start_date = (datetime.now() - timedelta(days=365*5)).date()
    
    try:
        symbols_to_fetch = TICKERS + ['SPY']
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols_to_fetch,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start_date,
            end=end_date,
            adjustment='all'
        )
        
        bars = data_client.get_stock_bars(request_params)
        
        # 1. Build dictionary of Series indexed by date
        data_dict = {}
        for symbol in symbols_to_fetch:
            if symbol in bars.data:
                bar_list = bars.data[symbol]
                # Use timestamp.date() as index to align everyone
                dates = [b.timestamp.date() for b in bar_list]
                closes = [b.close for b in bar_list]
                # Remove duplicates if any
                s = pd.Series(data=closes, index=dates)
                data_dict[symbol] = s[~s.index.duplicated(keep='last')]
        
        if not data_dict:
            raise ValueError("❌ No data returned from Alpaca")
            
        # 2. Combine into DataFrame (outer join by date)
        # This aligns AAPL (2000 days) with NewStock (50 days) correctly
        data = pd.DataFrame(data_dict)
        
        # 3. Drop assets that have too little history (e.g. < 100 days)
        # instead of chopping everyone else down.
        min_history = 100
        valid_cols = []
        for col in data.columns:
            # Count valid (non-NaN) points
            if data[col].count() >= min_history:
                valid_cols.append(col)
            else:
                # Optional: print dropped tickers to see who the culprit was
                pass
        
        if 'SPY' not in valid_cols:
             # Fallback: if SPY is missing/short, something is wrong with data feed
            raise ValueError("❌ SPY data is missing or too short!")
            
        data = data[valid_cols]
        
        # 4. Fill missing values
        data = data.ffill().bfill()
        
        print(f"✓ Downloaded {len(data)} trading days for {len(data.columns)} symbols")
        
        spy = data['SPY'].copy()
        assets = data.drop(columns=['SPY'])
        
        return assets, spy

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        raise RuntimeError(f"Failed to download data: {e}")

def get_current_price_from_alpaca(symbol):
    """
    Fetch current/latest price for a symbol from Alpaca.
    Returns float price or None on error.
    """
    try:
        request = StockLatestBarRequest(symbol_or_symbols=symbol)
        latest_bars = data_client.get_stock_latest_bar(request)

        if symbol in latest_bars:
            bar = latest_bars[symbol]
            price = float(bar.close)
            if price > 0:
                return price
        return None
    except Exception as e:
        print(f"⚠️ Could not get latest bar for {symbol}: {e}")
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=(datetime.now() - timedelta(days=5)).date(),
                end=datetime.now().date()
            )
            bars = data_client.get_stock_bars(request_params)
            if symbol in bars.data and len(bars.data[symbol]) > 0:
                price = float(bars.data[symbol][-1].close)
                if price > 0:
                    print(f"  ✓ Using fallback price for {symbol}: ${price:.2f}")
                    return price
        except Exception as e2:
            print(f"  ⚠️ Fallback also failed: {e2}")
        return None

# ==========================================
# 4. LIVE TRADING LOGIC (NO TIMING)
# ==========================================
def cancel_all_open_orders():
    """Cancel all open orders to free up shares/cash."""
    try:
        print("Cancelling all open orders to free up shares/cash...")
        trading_client.cancel_orders()
        time.sleep(3)
    except Exception as e:
        print(f"⚠️ Error cancelling orders: {e}")

def rebalance_portfolio():
    print(f"\n[{datetime.now()}] Starting Rebalance Routine...")

    cancel_all_open_orders()

    # 1. Load Data from Alpaca
    try:
        assets, spy = get_data_from_alpaca()
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    valid_tickers = assets.columns.tolist()

    # 2. Train Models
    print("Training RLS models on recent history...")
    filters = {t: RLSFilter(n=5, lam=CONFIG['rls_memory']) for t in valid_tickers}

    lookback = 200
    if len(assets) > lookback:
        train_assets = assets.iloc[-lookback:]
        train_spy = spy.iloc[-lookback:]
    else:
        train_assets = assets
        train_spy = spy

    for i in range(60, len(train_assets) - 1):
        for t in valid_tickers:
            hist = train_assets[t].iloc[:i+1]
            spy_hist = train_spy.iloc[:i+1]
            x = get_features(hist, spy_hist)
            y_actual = train_assets[t].pct_change().iloc[i+1]
            if x is not None and np.isfinite(y_actual):
                filters[t].adapt(x, y_actual)

    # 3. Generate Predictions
    print("Generating predictions...")
    scores = {}
    for t in valid_tickers:
        x = get_features(assets[t], spy)
        if x is not None:
            scores[t] = filters[t].predict(x)

    print(f"Generated scores for {len(scores)} assets")

    # 4. Select Portfolio
    bullish = {k: v for k, v in scores.items() if v > 0}
    target_weights = {}

    if not bullish:
        print("No bullish signals. Going 100% SHY (Cash proxy).")
        if 'SHY' in valid_tickers:
            target_weights = {'SHY': 1.0}
        else:
            print("⚠️ Warning: SHY not available. Portfolio will be cash.")
            target_weights = {}
    else:
        sorted_bulls = sorted(bullish.items(),
                              key=lambda x: (x[1], x[0]),
                              reverse=True)
        top_picks = sorted_bulls[:CONFIG['top_n']]
        n = len(top_picks)
        raw_weights = [(sym, n - rank + 1)
                       for rank, (sym, score) in enumerate(top_picks, start=1)]
        total_raw = sum(w for _, w in raw_weights)
        target_weights = {sym: w / total_raw for sym, w in raw_weights}

    print(f"Target Portfolio: {list(target_weights.keys())}")
    print(f"Target Weights: {target_weights}")

    # 5. Execute Trades
    execute_alpaca_trades(target_weights, assets)
    print("Rebalance Complete.")

def execute_alpaca_trades(target_weights, assets):
    """
    Sync Alpaca account with target weights using CURRENT Alpaca prices.
    """
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)

        print(f"Account Equity: ${equity:,.2f}")
        print(f"Buying Power: ${buying_power:,.2f}")
    except Exception as e:
        print(f"❌ Could not fetch account info: {e}")
        return

    # Get current positions
    try:
        positions = trading_client.get_all_positions()
        current_holdings = {p.symbol: float(p.qty) for p in positions}
        print(f"Current positions: {current_holdings}")
    except Exception as e:
        print(f"❌ Could not fetch positions: {e}")
        return

    # PHASE 1: SELL exits
    for symbol, qty in current_holdings.items():
        if symbol not in target_weights:
            print(f"\n📉 Selling {symbol} (Not in target)...")
            try:
                current_price = get_current_price_from_alpaca(symbol)
                if current_price is None or current_price <= 0:
                    print(f"⚠️ Could not get valid price for {symbol}. Skipping.")
                    continue

                sell_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=True,
                    limit_price=current_price*0.98    # or a slightly higher price to ensure fill
                )
                order = trading_client.submit_order(sell_request)
                print(f"✓ Submitted SELL: {int(qty)} {symbol} @ ~${current_price:.2f}")
            except Exception as e:
                print(f"❌ Error selling {symbol}: {e}")

    print("\n⏳ Waiting for sell orders to process...")
    time.sleep(3)

    # PHASE 2: BUY new positions
    for symbol, weight in target_weights.items():
        target_val = equity * weight

        if target_val < 1:
            print(f"⏭️  Skipping {symbol} (allocation too small: ${target_val:.2f})")
            continue

        try:
            current_price = get_current_price_from_alpaca(symbol)
            if current_price is None or current_price <= 0:
                print(f"⚠️ Could not get valid price for {symbol}. Skipping.")
                continue

            # --- FRACTIONAL FIRST ---
            # Try to buy using notional (fractional shares)
            notional = round(target_val, 2)
            if notional < 1:
                print(f"⏭️  Skipping {symbol} (notional < $1: ${notional:.2f})")
                continue

            limit_price = round(current_price*1.02, 2)

            try:
                # Fractional LIMIT order (works in RTH & extended if enabled)
                buy_request = LimitOrderRequest(
                    symbol=symbol,
                    notional=notional,          # key: use notional for fractional
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=True,        # or False if you only want RTH
                    limit_price=limit_price
                )
                order = trading_client.submit_order(buy_request)
                print(f"✓ Fractional BUY submitted: ${notional:.2f} of {symbol} @ ~${limit_price:.2f}")
                continue  # done with this symbol if fractional worked

            except Exception as e_frac:
                print(f"⚠️ Fractional order failed for {symbol}: {e_frac}")
                # Fall back to whole shares below

            # --- FALLBACK: WHOLE SHARES ---
            qty = int(target_val // current_price)
            if qty < 1:
                print(f"⏭️  Skipping {symbol} (cannot afford 1 share at ${current_price:.2f})")
                continue

            # You can choose Market or Limit here; Market shown for simplicity
            buy_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
                # Do NOT set extended_hours=True here unless you switch to limit,
                # because extended hours + market is rejected.
            )
            order = trading_client.submit_order(buy_request)
            cost = qty * current_price
            remaining = target_val - cost

            print(
                f"✓ Whole-share BUY submitted: {qty} {symbol} @ ~${current_price:.2f} "
                f"(Cost: ${cost:.2f}, Remaining: ${remaining:.2f})"
            )

        except Exception as e:
            print(f"❌ Error buying {symbol}: {e}")

    print("\n✅ Trade execution complete!")

# ==========================================
# 5. MAIN (NO SCHEDULER / NO TIMING)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Alpaca Trading Bot Started (IMMEDIATE REBALANCE TEST)...")
    print("=" * 60)

    try:
        account = trading_client.get_account()
        print("✓ Connected to Alpaca")
        print(f"  Account: {account.account_number}")
        print(f"  Paper Trading: {PAPER}")
        print(f"  Equity: ${float(account.equity):,.2f}")
    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        exit(1)

    # DIRECTLY run one full rebalance now
    rebalance_portfolio()