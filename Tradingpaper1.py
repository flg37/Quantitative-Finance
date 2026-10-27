import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ------------------------------------------------------------
# 1. Configuration (Hardcoded Keys)
# ------------------------------------------------------------
API_KEY = ""
API_SECRET = ""

# Connect to Alpaca Paper Trading
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Universe (Removed BIL as per your model)
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'GLD', 'VNQ', 'XLE', 'XLF', 'XLK']

# Model Parameters
LOOKBACK_DAYS = 252
HALFLIFE = 21
N_FACTS = 4

# ------------------------------------------------------------
# 2. Model Helper Functions (Exact Match to Backtest)
# ------------------------------------------------------------
def get_ewm_stats(window_data_df, halflife=21):
    data_arr = window_data_df.values
    T, N = data_arr.shape
    alpha = np.log(0.5) / halflife
    weights = np.exp(alpha * np.arange(T)[::-1])
    weights /= weights.sum()
    
    expected_returns = (data_arr.T @ weights) * 252
    demeaned = data_arr - (data_arr.T @ weights)
    weighted_demeaned = demeaned * weights[:, np.newaxis]
    cov = (demeaned.T @ weighted_demeaned) * 252
    return expected_returns, cov

def get_denoised_cov(cov_matrix_df, n_facts):
    cov_vals = cov_matrix_df.values
    std = np.sqrt(np.diag(cov_vals))
    corr = cov_vals / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0)
    
    pca = PCA()
    pca.fit(corr)
    evals = pca.explained_variance_
    evecs = pca.components_.T
    evals_denoised = np.copy(evals)
    evals_denoised[n_facts:] = evals[n_facts:].mean()
    
    lambda_mat = np.diag(evals_denoised)
    denoised_corr = evecs.dot(lambda_mat).dot(evecs.T)
    np.fill_diagonal(denoised_corr, 1.0)
    
    denoised_cov = np.outer(std, std) * denoised_corr
    return pd.DataFrame(denoised_cov, index=cov_matrix_df.index, columns=cov_matrix_df.columns)

def get_max_sharpe_weights(cov_matrix, expected_returns):
    num_assets = len(cov_matrix)
    cov_matrix = cov_matrix + np.eye(num_assets) * 1e-6
    
    def neg_sharpe(weights):
        p_ret = np.sum(weights * expected_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if p_vol <= 1e-8: return 0
        return -(p_ret / p_vol)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    try:
        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-5)
        return pd.Series(result.x, index=cov_matrix.index)
    except:
        return pd.Series(init_guess, index=cov_matrix.index)

# ------------------------------------------------------------
# 3. Live Trading Logic
# ------------------------------------------------------------
def compute_target_weights():
    """
    Downloads data, runs the model, and returns target weights for today.
    """
    print(f"Fetching data for {len(TICKERS)} tickers...")
    # Fetch ample history to ensure we have 252 trading days
    data = yf.download(TICKERS, period="2y", auto_adjust=True)['Close']
    returns = np.log(data / data.shift(1)).dropna()
    
    # Slice the last 'window' days
    if len(returns) < LOOKBACK_DAYS:
        raise ValueError(f"Not enough data history. Needed {LOOKBACK_DAYS}, got {len(returns)}")
        
    window_data = returns.iloc[-LOOKBACK_DAYS:]
    
    # 1. Stats
    ewm_ret, ewm_cov = get_ewm_stats(window_data, halflife=HALFLIFE)
    ewm_cov_df = pd.DataFrame(ewm_cov, index=TICKERS, columns=TICKERS)
    ewm_ret_series = pd.Series(ewm_ret, index=TICKERS)
    
    # 2. Denoise
    denoised_cov = get_denoised_cov(ewm_cov_df, n_facts=N_FACTS)
    
    # 3. Filter (Trend)
    ewm_ret_series[ewm_ret_series < 0] = 0
    
    # 4. Optimize
    # Safety check: if all returns are negative, default to equal weights or cash (here equal weights)
    if ewm_ret_series.sum() == 0:
        print("Warning: All expected returns negative. Defaulting to Equal Weight.")
        return pd.Series(1.0/len(TICKERS), index=TICKERS)
        
    w_opt = get_max_sharpe_weights(denoised_cov, ewm_ret_series)
    return w_opt

def execute_rebalance(target_weights):
    """
    Calculates orders based on target weights and executes them via Alpaca.
    """
    # 1. Get Account Info
    account = trading_client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    
    print(f"Account Equity: ${equity:.2f}")
    print(f"Buying Power:   ${buying_power:.2f}")
    
    # 2. Get Current Positions
    positions = trading_client.get_all_positions()
    current_pos = {p.symbol: float(p.market_value) for p in positions}
    
    # 3. Calculate Target Value for each asset
    # We use equity * 0.98 to leave a 2% cash buffer for safety/slippage
    target_equity = equity * 0.98
    target_values = target_weights * target_equity
    
    # 4. Generate Orders
    print("\n--- Generating Orders ---")
    orders = []
    
    for symbol in TICKERS:
        current_val = current_pos.get(symbol, 0.0)
        target_val = target_values.get(symbol, 0.0)
        diff = target_val - current_val
        
        # Only trade if difference is > $10 to avoid dust trades
        if abs(diff) < 10:
            continue
            
        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        notional = round(abs(diff), 2)
        
        # BUYING POWER CHECK
        if side == OrderSide.BUY:
            if notional > buying_power:
                print(f"WARNING: Insufficient BP for {symbol}. Clipping {notional} -> {buying_power}")
                notional = round(buying_power, 2)
            # Update local tracking of BP so next orders don't fail
            buying_power -= notional
        
        if notional > 1.0: # Minimum order size $1
            orders.append({'symbol': symbol, 'notional': notional, 'side': side})

    # 5. Execute Orders
    print(f"Sending {len(orders)} orders...")
    for order_data in orders:
        try:
            print(f"Submitting {order_data['side']} ${order_data['notional']} {order_data['symbol']}")
            req = MarketOrderRequest(
                symbol=order_data['symbol'],
                notional=order_data['notional'],
                side=order_data['side'],
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(req)
            time.sleep(0.5) # Rate limit protection
        except Exception as e:
            print(f"ERROR submitting order for {order_data['symbol']}: {e}")

# ------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    print("--- Starting Daily Rebalance ---")
    
    try:
        # 1. Compute Weights
        weights = compute_target_weights()
        print("\nTarget Weights:")
        print(weights[weights > 0.001].round(4))
        
        # 2. Execute
        execute_rebalance(weights)
        print("\n--- Rebalance Complete ---")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR during rebalance: {e}")