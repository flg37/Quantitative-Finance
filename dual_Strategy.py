"""
OANDA Flow Scanner v7.3 -- Z-Reversion Exit Edition
Weekend-aware + stack-size deleverage + weekend-idle edition.

MT5/Darwinex leg removed -- OANDA-only version of Dual_Strategy8.py.

Changes vs original:
  1. Dual OANDA clients: live for trading, practice for position-book data.
  2. hours_held uses trading_hours_between() (excludes Fri 21:00 -> Sun 21:00 UTC).
  3. Strict-FIFO stack-size deleverage via DELEVERAGE_EXIT_Z.
  4. Phone summary check tolerates a half-scan-interval of timing jitter.
  5. Whole scan loop idles (Telegram only) when _is_market_closed() is True.
"""

import os
import json
import time
import logging
from collections import deque
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

import oandapyV20
from oandapyV20.endpoints import instruments, accounts, orders, trades, pricing


# ==========================================
# CONFIGURATION
# ==========================================
# OANDA LIVE (trading, account, price data, candles)
ACCESS_TOKEN = ""        # <-- live token
ACCOUNT_ID   = ""        # <-- live account ID
ENVIRONMENT  = "live"

# OANDA PRACTICE (used ONLY for retail position-book sentiment data,
# which is restricted on live accounts. The position book is market-wide,
# so practice returns identical data to what live would return.)
DATA_ACCESS_TOKEN = ""  # 
DATA_ENVIRONMENT  = "practice"


TELEGRAM_ENABLED = True
TELEGRAM_CHAT_ID = ""
TELEGRAM_BOT_TOKEN = ""
bot_enabled = True
last_telegram_update_id = None


PHONE_SUMMARY_INTERVAL_SECONDS = 20 * 60
last_phone_summary_time = 0.0

PAIRS = ["AUD_USD", "NZD_USD", "GBP_USD"]
CORRELATION_GROUPS = [["GBP_USD", "AUD_USD", "NZD_USD"]]

MAX_CONCURRENT_TRADES_PER_PAIR = 5
MAX_TOTAL_CORRELATED_TRADES    = 20
MAX_ACCOUNT_RISK_PCT           = 0.02

SAFETY_SL_DAILY_ATR_MULT = 3.0
SAFETY_TP_DAILY_ATR_MULT = 5.0

SCAN_INTERVAL          = 600
TRADE_COOLDOWN_SECONDS = 3600
CONFIRMATION_SCANS     = 2

DEFAULT_PARAMS = {"entry_z": 1.5, "exit_z": 0.5, "max_hold_hours": 72}

DELEVERAGE_EXIT_Z = {
    5: 0.75,
    4: 0.57,
    3: 0.45,
    2: 0.40,
    1: 0.35,
}

OANDA_TAG = "sentiment_v7"


# =========================================================================
# LOGGING
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("dual_broker_scanner.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# =========================================================================
# STATE
# =========================================================================
oanda_trade_counts = {t: 0 for t in PAIRS}
sentiment_streak   = {t: 0 for t in PAIRS}

last_trade_time    = {t: 0.0 for t in PAIRS}

pair_params = {t: dict(DEFAULT_PARAMS) for t in PAIRS}

dynamic_baselines = {
    t: {
        "long_q":  deque(maxlen=720),
        "short_q": deque(maxlen=720),
        "long_mean": 50.0, "long_std": 1.0,
        "short_mean": 50.0, "short_std": 1.0,
        "last_update_key": None,
    }
    for t in PAIRS
}

client      = oandapyV20.API(access_token=ACCESS_TOKEN,      environment=ENVIRONMENT)
data_client = oandapyV20.API(access_token=DATA_ACCESS_TOKEN, environment=DATA_ENVIRONMENT)

script_dir = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# HELPERS: WEEKEND-AWARE HOLD TIME + DELEVERAGE LOOKUP
# =========================================================================
def _is_market_closed(dt):
    """
    Forex weekend window:
      - All of Saturday              (weekday == 5)
      - Sunday before 21:00 UTC      (weekday == 6, hour < 21)
      - Friday at or after 21:00 UTC (weekday == 4, hour >= 21)
    """
    dow = dt.weekday()
    if dow == 5:                       return True
    if dow == 6 and dt.hour < 21:      return True
    if dow == 4 and dt.hour >= 21:     return True
    return False


def trading_hours_between(start_utc, end_utc):
    """
    Hours between two UTC datetimes, EXCLUDING the forex weekend closure
    (Fri 21:00 UTC -> Sun 21:00 UTC). Handles fractional hours correctly.
    """
    if end_utc <= start_utc:
        return 0.0

    one_hour = timedelta(hours=1)
    total = 0.0
    cur = start_utc

    while cur < end_utc:
        next_top = cur.replace(minute=0, second=0, microsecond=0) + one_hour
        slice_end = min(next_top, end_utc)
        if not _is_market_closed(cur):
            total += (slice_end - cur).total_seconds() / 3600
        cur = slice_end

    return total


def dynamic_exit_z(count, fallback):
    """Candidate trade's exit_z based on current stack size."""
    return DELEVERAGE_EXIT_Z.get(count, fallback)


# =========================================================================
# BOOT SEQUENCE
# =========================================================================
def run_boot_sequence():
    log.info("=== Initiating Boot Sequence ===")

    json_path = os.path.join(script_dir, "optimal_params.json")
    try:
        with open(json_path, "r") as f:
            loaded = json.load(f)
        for t in PAIRS:
            if t in loaded:
                p = loaded[t]
                pair_params[t] = {
                    "entry_z":        float(p.get("entry_z",        DEFAULT_PARAMS["entry_z"])),
                    "exit_z":         float(p.get("exit_z",         DEFAULT_PARAMS["exit_z"])),
                    "max_hold_hours": int  (p.get("max_hold_hours", DEFAULT_PARAMS["max_hold_hours"])),
                }
        log.info(f"  [JSON] Loaded per-pair params from optimal_params.json")
        for t in PAIRS:
            p = pair_params[t]
            log.info(f"         {t}: entry_z={p['entry_z']}  "
                     f"exit_z={p['exit_z']}  max_hold={p['max_hold_hours']}h")
    except Exception as e:
        log.warning(f"  [JSON] Could not load {json_path}: {e}")
        log.warning(f"         Using defaults for all pairs: {DEFAULT_PARAMS}")

    log.info(f"  [DELEVERAGE] candidate exit_z schedule: {DELEVERAGE_EXIT_Z}")

    for ticker in PAIRS:
        for fname in [f"{ticker}_24M_flow_history.csv",
                      f"{ticker}_flow_history.csv"]:
            csv_file = os.path.join(script_dir, fname)
            if not os.path.exists(csv_file):
                continue
            try:
                df = pd.read_csv(csv_file, parse_dates=["Date"]).set_index("Date")
                last_data = df.tail(720)
                b = dynamic_baselines[ticker]
                b["long_q"].extend(last_data["Long_Pct"].tolist())
                b["short_q"].extend(last_data["Short_Pct"].tolist())
                b["long_mean"]  = float(np.mean(b["long_q"]))
                b["long_std"]   = float(max(np.std(b["long_q"]),  0.1))
                b["short_mean"] = float(np.mean(b["short_q"]))
                b["short_std"]  = float(max(np.std(b["short_q"]), 0.1))
                b["last_update_key"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H")
                log.info(f"  [MEMORY] {ticker} baseline initialised from {fname}")
                break
            except Exception as e:
                log.error(f"  [BOOT ERROR] {ticker}: {e}")


# =========================================================================
# BROKER STATE SYNC
# =========================================================================
def sync_broker_states():
    try:
        r = trades.OpenTrades(accountID=ACCOUNT_ID)
        client.request(r)
        for t in PAIRS:
            oanda_trade_counts[t] = 0
        untagged = 0
        for trade in r.response.get("trades", []):
            instrument = trade["instrument"]
            if instrument not in oanda_trade_counts:
                continue
            oanda_trade_counts[instrument] += 1

            ext = trade.get("clientExtensions", {})
            if ext.get("tag") != OANDA_TAG:
                untagged += 1

        if untagged > 0:
            log.warning(f"  [OANDA SYNC] {untagged} open trade(s) on configured pairs "
                        f"lack '{OANDA_TAG}' tag -- counted for capacity but NOT "
                        f"managed by Z-exit. Run tag_existing_oanda_trades.py to fix.")
    except Exception as e:
        log.error(f"  [OANDA SYNC ERROR] {e}")


# =========================================================================
# DATA HELPERS
# =========================================================================
def get_oanda_balance():
    oanda_bal = None
    try:
        r = accounts.AccountSummary(ACCOUNT_ID)
        client.request(r)
        oanda_bal = float(r.response["account"]["balance"])
    except Exception:
        pass
    return oanda_bal


def get_current_price_oanda(ticker, side="mid"):
    try:
        r = pricing.PricingInfo(accountID=ACCOUNT_ID, params={"instruments": ticker})
        client.request(r)
        px = r.response["prices"][0]
        bid, ask = float(px["bids"][0]["price"]), float(px["asks"][0]["price"])
        if side == "bid": return bid
        if side == "ask": return ask
        return round((bid + ask) / 2, 5)
    except Exception:
        return None


def get_gbpusd_rate():
    """Fetch live GBP/USD mid from OANDA. Raises on failure."""
    r = pricing.PricingInfo(accountID=ACCOUNT_ID, params={"instruments": "GBP_USD"})
    client.request(r)
    px = r.response["prices"][0]
    return (float(px["bids"][0]["price"]) + float(px["asks"][0]["price"])) / 2


def get_daily_atr(ticker, period=14):
    try:
        r = instruments.InstrumentsCandles(
            instrument=ticker,
            params={"count": period + 1, "granularity": "D", "price": "M"},
        )
        client.request(r)
        candles = r.response.get("candles", [])
        trs = []
        for i in range(1, len(candles)):
            h, l = float(candles[i]["mid"]["h"]), float(candles[i]["mid"]["l"])
            pc   = float(candles[i - 1]["mid"]["c"])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs) / len(trs) if trs else None
    except Exception:
        return None


def get_retail_sentiment(ticker):
    try:
        r = instruments.InstrumentsPositionBook(instrument=ticker)
        data_client.request(r)

        buckets = r.response.get("positionBook", {}).get("buckets", [])

        if not buckets:
            log.warning(f"  [SENTIMENT WARNING] {ticker}: positionBook returned no buckets")
            return None, None

        long_pct  = sum(float(b.get("longCountPercent",  0)) for b in buckets)
        short_pct = sum(float(b.get("shortCountPercent", 0)) for b in buckets)

        return long_pct, short_pct

    except Exception as e:
        log.error(f"  [SENTIMENT ERROR] {ticker}: {e}")
        return None, None


# =========================================================================
# POSITION SIZING
# =========================================================================
def calculate_oanda_size(stop_dist, oanda_bal, ticker):
    oanda_units = 0

    if oanda_bal and oanda_bal > 0 and stop_dist > 0:
        try:
            gbpusd = get_gbpusd_rate()
        except Exception as e:
            log.error(f"  [SIZING] {ticker} OANDA leg skipped -- GBP/USD fetch failed: {e}")
            gbpusd = None

        if gbpusd is not None:
            risk_usd   = oanda_bal * MAX_ACCOUNT_RISK_PCT * gbpusd
            calculated = int(risk_usd / stop_dist)
            if calculated < 1:
                log.warning(f"  [SIZING] {ticker} OANDA units {calculated} below minimum -- skipping")
            else:
                oanda_units = calculated

    return oanda_units


# =========================================================================
# CORRELATION & CAPACITY CHECKS
# =========================================================================
def correlated_group_total(broker_counts, ticker):
    for group in CORRELATION_GROUPS:
        if ticker in group:
            return sum(broker_counts.get(t, 0) for t in group)
    return broker_counts.get(ticker, 0)


def can_open_oanda(ticker):
    if oanda_trade_counts[ticker] >= MAX_CONCURRENT_TRADES_PER_PAIR:
        return False
    if correlated_group_total(oanda_trade_counts, ticker) >= MAX_TOTAL_CORRELATED_TRADES:
        return False
    return True


# =========================================================================
# ORDER EXECUTION
# =========================================================================
def execute_oanda(ticker, units, sl_dist, tp_dist):
    if units == 0:
        return False
    try:
        prec = 3 if "JPY" in ticker else 5
        ts   = int(time.time())
        data = {
            "order": {
                "type": "MARKET",
                "instrument": ticker,
                "units": str(units),
                "timeInForce": "FOK",
                "stopLossOnFill":   {"distance": f"{sl_dist:.{prec}f}"},
                "takeProfitOnFill": {"distance": f"{tp_dist:.{prec}f}"},
                "clientExtensions": {
                    "id":  f"sentiment_order_{ticker}_{ts}_{abs(units)}",
                    "tag": OANDA_TAG,
                },
                "tradeClientExtensions": {
                    "id":  f"sentiment_trade_{ticker}_{ts}_{abs(units)}",
                    "tag": OANDA_TAG,
                },
            }
        }
        r = orders.OrderCreate(accountID=ACCOUNT_ID, data=data)
        client.request(r)
        if r.response.get("orderFillTransaction"):
            log.info(f"  >>> [OANDA OPEN] {'LONG' if units > 0 else 'SHORT'} "
                     f"{ticker} | {abs(units)} units")
            return True
    except Exception as e:
        log.error(f"  >>> [OANDA REJECTED] {e}")
    return False


# =========================================================================
# Z-EXIT TRADE MANAGEMENT
# =========================================================================
def close_oanda_trade(trade_id, ticker, side_str, reason):
    try:
        r = trades.TradeClose(accountID=ACCOUNT_ID, tradeID=str(trade_id))
        client.request(r)
        log.info(f"  >>> [OANDA EXIT] {side_str} {ticker} #{trade_id} | {reason}")
        return True
    except Exception as e:
        log.error(f"  >>> [OANDA CLOSE ERROR] #{trade_id}: {e}")
        return False


def manage_open_trades(now_utc, current_z):
    """
    Strict-FIFO z-exit with CASCADING stack-size deleverage:
      - Each scan, the oldest trade per pair is the candidate.
        Its threshold = DELEVERAGE_EXIT_Z[count].
      - If the candidate's threshold is met, it closes. Then the next-
        oldest becomes the new candidate at the new (tighter) threshold,
        which is re-checked against the SAME current Z. The cascade
        continues until the threshold is no longer met or no trades remain.
      - max_hold_hours timeout applies to EVERY trade independently
        (not gated by candidate status), checked before the z-exit cascade.

    hours_held EXCLUDES the forex weekend closure window.
    """
    closed_count = 0

    try:
        r = trades.OpenTrades(accountID=ACCOUNT_ID)
        client.request(r)

        oanda_stacks = {}
        for trade in r.response.get("trades", []):
            ext = trade.get("clientExtensions", {})
            if ext.get("tag") != OANDA_TAG:
                continue
            ticker = trade["instrument"]
            if ticker not in PAIRS or ticker not in current_z:
                continue
            oanda_stacks.setdefault(ticker, []).append(trade)

        for ticker, stack in oanda_stacks.items():
            stack.sort(key=lambda t: pd.to_datetime(t["openTime"]))  # oldest first

            params      = pair_params[ticker]
            max_hold    = params["max_hold_hours"]
            base_exit_z = params["exit_z"]
            l_z, s_z    = current_z[ticker]

            # Pass 1: max-hold timeouts apply to every trade independently
            remaining = []
            for trade in stack:
                units      = float(trade["currentUnits"])
                is_long    = units > 0
                entry_time = pd.to_datetime(trade["openTime"]).to_pydatetime()
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                hours_held = trading_hours_between(entry_time, now_utc)

                if hours_held >= max_hold:
                    reason = f"timeout {hours_held:.1f}h>={max_hold}h"
                    if close_oanda_trade(trade["id"], ticker,
                                         "LONG" if is_long else "SHORT", reason):
                        closed_count += 1
                    else:
                        remaining.append(trade)   # close failed, keep tracking
                else:
                    remaining.append(trade)

            # Pass 2: cascading candidate z-exit
            while remaining:
                candidate = remaining[0]
                count     = len(remaining)
                cand_exit_z = dynamic_exit_z(count, base_exit_z)

                units   = float(candidate["currentUnits"])
                is_long = units > 0

                triggered, reason = False, ""
                if is_long and s_z <= cand_exit_z:
                    triggered = True
                    reason = f"Short_Z={s_z:.2f}<={cand_exit_z} (stack={count})"
                elif (not is_long) and l_z <= cand_exit_z:
                    triggered = True
                    reason = f"Long_Z={l_z:.2f}<={cand_exit_z} (stack={count})"

                if not triggered:
                    break   # threshold no longer met, cascade ends

                if close_oanda_trade(candidate["id"], ticker,
                                     "LONG" if is_long else "SHORT", reason):
                    closed_count += 1
                    remaining.pop(0)
                else:
                    break   # close failed, abort cascade to avoid infinite loop
    except Exception as e:
        log.error(f"  [OANDA MANAGE ERROR] {e}")

    if closed_count:
        log.info(f"  [MANAGE] Closed {closed_count} position(s) this scan.")
    return closed_count


# =========================================================================
# TELEGRAM
# =========================================================================
def send_telegram(message):
    if not TELEGRAM_ENABLED:
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }

        response = requests.post(url, data=payload, timeout=10)

        if response.status_code != 200:
            log.error(f"  [TELEGRAM ERROR] {response.text}")
            return False

        return True

    except Exception as e:
        log.error(f"  [TELEGRAM ERROR] {e}")
        return False


def check_telegram_commands():
    global bot_enabled, last_telegram_update_id

    if not TELEGRAM_ENABLED:
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"

        params = {}
        if last_telegram_update_id is not None:
            params["offset"] = last_telegram_update_id + 1

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            log.error(f"  [TELEGRAM CONTROL ERROR] {response.text}")
            return

        updates = response.json().get("result", [])

        for update in updates:
            last_telegram_update_id = update["update_id"]

            message = update.get("message", {})
            chat_id = str(message.get("chat", {}).get("id", ""))

            if chat_id != str(TELEGRAM_CHAT_ID):
                log.warning(f"  [TELEGRAM CONTROL] Ignored command from unknown chat_id={chat_id}")
                continue

            text = message.get("text", "").strip().lower()

            if text == "/stopbot":
                bot_enabled = False
                send_telegram("Bot paused. No new trades will be opened.")

            elif text == "/startbot":
                bot_enabled = True
                send_telegram("Bot resumed. Trading logic is active again.")

            elif text == "/status":
                status = "ACTIVE" if bot_enabled else "PAUSED"
                send_telegram(f"Bot status: {status}")

    except Exception as e:
        log.error(f"  [TELEGRAM CONTROL ERROR] {e}")


# =========================================================================
# MAIN SCANNER
# =========================================================================
def run_flow_scanner():
    run_boot_sequence()
    log.info(f"\n=== OANDA Flow Scanner v7.3 -- Z-EXIT (Weekend + Deleverage + Idle) ===")
    log.info(f"  Per-pair limit: {MAX_CONCURRENT_TRADES_PER_PAIR} trades  |  "
             f"Group total cap: {MAX_TOTAL_CORRELATED_TRADES} across {CORRELATION_GROUPS[0]}")
    log.info(f"  Safety stops: SL={SAFETY_SL_DAILY_ATR_MULT}x / TP={SAFETY_TP_DAILY_ATR_MULT}x DAILY ATR")
    log.info(f"  Cooldown: {TRADE_COOLDOWN_SECONDS//60}m per pair")
    next_scan_time = time.time()
    while True:
        try:
            global last_phone_summary_time

            now_utc = datetime.now(timezone.utc)
            scan_unix = now_utc.timestamp()

            # Always poll Telegram so /status, /stopbot, /startbot work even on weekends.
            check_telegram_commands()

            if _is_market_closed(now_utc):
                log.info(f"  [MARKET CLOSED] {now_utc.strftime('%a %H:%M UTC')} "
                         f"-- skipping scan; no sentiment fetch, no trade activity.")
            else:
                scan_summary_lines = []

                sync_broker_states()
                oanda_bal = get_oanda_balance()

                log.info(f"--- Scan: {now_utc.strftime('%H:%M:%S')} UTC | "
                         f"OANDA: GBP {oanda_bal or 0:,.2f} ---")

                scan_summary_lines.append(
                    f"Flow Scanner Update - {now_utc.strftime('%H:%M:%S')} UTC"
                )
                scan_summary_lines.append(
                    f"OANDA: GBP {oanda_bal or 0:,.2f}"
                )
                scan_summary_lines.append("")

                current_z = {}
                for ticker in PAIRS:
                    time.sleep(1)

                    l_pct, s_pct = get_retail_sentiment(ticker)
                    if l_pct is None:
                        log.warning(f"  [SKIP] {ticker}: no sentiment data, CSV not updated")
                        continue

                    b   = dynamic_baselines[ticker]
                    l_z = (l_pct - b["long_mean"])  / b["long_std"]
                    s_z = (s_pct - b["short_mean"]) / b["short_std"]
                    current_z[ticker] = (l_z, s_z)

                    hour_key = now_utc.strftime("%Y-%m-%d %H")

                    if hour_key != b["last_update_key"]:
                        b["long_q"].append(l_pct)
                        b["short_q"].append(s_pct)
                        b["long_mean"]  = float(np.mean(b["long_q"]))
                        b["long_std"]   = float(max(np.std(b["long_q"]),  0.1))
                        b["short_mean"] = float(np.mean(b["short_q"]))
                        b["short_std"]  = float(max(np.std(b["short_q"]), 0.1))
                        b["last_update_key"] = hour_key

                        csv_path = os.path.join(script_dir, f"{ticker}_24M_flow_history.csv")
                        try:
                            with open(csv_path, "a") as f:
                                if f.tell() == 0:
                                    f.write("Date,Long_Pct,Short_Pct\n")
                                f.write(f"{now_utc.strftime('%Y-%m-%d %H:00:00')},{l_pct},{s_pct}\n")
                        except Exception as e:
                            log.error(f"  [CSV ERROR] {e}")

                closed = manage_open_trades(now_utc, current_z)
                if closed:
                    sync_broker_states()

                if not bot_enabled:
                    log.info("  [CONTROL] Bot paused. Skipping new trade entries.")
                    scan_summary_lines.append("Bot status: PAUSED - no new entries.")
                else:
                    for ticker in PAIRS:
                        if ticker not in current_z:
                            continue

                        l_z, s_z = current_z[ticker]
                        params   = pair_params[ticker]
                        target_z = params["entry_z"]

                        cur_unix = time.time()
                        cd_o = max(0, TRADE_COOLDOWN_SECONDS - (cur_unix - last_trade_time[ticker]))

                        o_count = oanda_trade_counts[ticker]
                        grp_o   = correlated_group_total(oanda_trade_counts, ticker)

                        cd_log = ""
                        if cd_o > 0:
                            cd_log = f" | CD: {int(cd_o/60)}m"

                        log.info(f"  {ticker:<8} | L_Z: {l_z:>5.2f} S_Z: {s_z:>5.2f} | "
                                 f"target_Z: {target_z} | "
                                 f"Pair: {o_count} | "
                                 f"Group: {grp_o}{cd_log}")

                        scan_summary_lines.append(
                                f"{ticker}: L_Z={l_z:.2f}, S_Z={s_z:.2f}, "
                                f"Target={target_z}, Pair={o_count}, "
                                f"Group={grp_o}"
                            )

                        if cd_o > 0:
                            sentiment_streak[ticker] = 0
                            continue

                        can_o = can_open_oanda(ticker)

                        if not can_o:
                            sentiment_streak[ticker] = 0
                            continue

                        if l_z >= target_z and s_z >= target_z:
                            sentiment_streak[ticker] = 0
                        elif l_z >= target_z:
                            sentiment_streak[ticker] = min(sentiment_streak[ticker] - 1, -1)
                        elif s_z >= target_z:
                            sentiment_streak[ticker] = max(sentiment_streak[ticker] + 1, 1)
                        else:
                            sentiment_streak[ticker] = 0

                        streak = sentiment_streak[ticker]

                        # --- SHORT signal (retail crowded long -> fade) ---------
                        if streak <= -CONFIRMATION_SCANS:
                            daily_atr = get_daily_atr(ticker)
                            if daily_atr:
                                sl_dist = daily_atr * SAFETY_SL_DAILY_ATR_MULT
                                tp_dist = daily_atr * SAFETY_TP_DAILY_ATR_MULT
                                o_units = calculate_oanda_size(sl_dist, oanda_bal, ticker)

                                fired = False
                                if can_o and cd_o == 0 and o_units > 0:
                                    if execute_oanda(ticker, -o_units, sl_dist, tp_dist):
                                        last_trade_time[ticker] = time.time()
                                        fired = True

                                if fired:
                                    sentiment_streak[ticker] = 0

                        # --- LONG signal (retail crowded short -> fade) ---------
                        elif streak >= CONFIRMATION_SCANS:
                            daily_atr = get_daily_atr(ticker)
                            if daily_atr:
                                sl_dist = daily_atr * SAFETY_SL_DAILY_ATR_MULT
                                tp_dist = daily_atr * SAFETY_TP_DAILY_ATR_MULT
                                o_units = calculate_oanda_size(sl_dist, oanda_bal, ticker)

                                fired = False
                                if can_o and cd_o == 0 and o_units > 0:
                                    if execute_oanda(ticker, o_units, sl_dist, tp_dist):
                                        last_trade_time[ticker] = time.time()
                                        fired = True

                                if fired:
                                    sentiment_streak[ticker] = 0

                if scan_unix - last_phone_summary_time >= PHONE_SUMMARY_INTERVAL_SECONDS - (SCAN_INTERVAL / 2):
                    if scan_summary_lines:
                        if send_telegram("\n".join(scan_summary_lines)):
                            last_phone_summary_time = scan_unix

                log.info("-" * 65)
        except Exception as e:
            log.exception(f"CRITICAL ERROR: {e}")

        next_scan_time += SCAN_INTERVAL
        sleep_time = max(0, next_scan_time - time.time())

        if sleep_time > 0:
            log.info(f"  [TIMER] Sleeping {sleep_time:.1f}s until next scheduled scan.")
            time.sleep(sleep_time)
        else:
            log.warning(f"  [TIMER] Scan overran schedule by {-sleep_time:.1f}s. Starting next scan immediately.")


if __name__ == "__main__":
    run_flow_scanner()