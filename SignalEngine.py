import numpy as np
import pandas as pd

def generate_signal(rates: list) -> dict:
    """
    Accepts OHLCV from MT5. Returns full trade plan with entry, TP1-3, SL, and signal reasons.
    """

    df = pd.DataFrame(rates)
    if df.empty or len(df) < 200:
        return {
            "signal": "NEUTRAL",
            "reason": ["Not enough data"],
            "entry": None,
            "tp_levels": [],
            "sl": None,
            "rsi": None,
            "trend": "unknown",
            "fvg_count": 0
        }

    reasons = []
    signal = "NEUTRAL"

    # === RSI ===
    df['diff'] = df['close'].diff()
    gain = df['diff'].clip(lower=0)
    loss = -df['diff'].clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    rsi_now = df['rsi'].iloc[-1]

    # === MA Trend ===
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    trend = "unknown"
    if df['ma50'].iloc[-1] > df['ma200'].iloc[-1]:
        trend = "UP"
        reasons.append("MA50 > MA200 → Uptrend")
    elif df['ma50'].iloc[-1] < df['ma200'].iloc[-1]:
        trend = "DOWN"
        reasons.append("MA50 < MA200 → Downtrend")

    # === Candle Bias ===
    last = df.iloc[-1]
    entry_price = last['close']

    if rsi_now < 30 and trend == "UP":
        signal = "BUY"
        reasons.append("RSI oversold + Uptrend")
    elif rsi_now > 70 and trend == "DOWN":
        signal = "SELL"
        reasons.append("RSI overbought + Downtrend")
    else:
        reasons.append("RSI/trend mismatch")

    # === Trade Plan Generation ===
    tp_levels = []
    sl = None
    pip_step = 0.002  # Example TP gap, adjustable per pair

    if signal == "BUY":
        tp_levels = [round(entry_price + pip_step * i, 5) for i in range(1, 4)]
        sl = round(entry_price - pip_step * 1.5, 5)
    elif signal == "SELL":
        tp_levels = [round(entry_price - pip_step * i, 5) for i in range(1, 4)]
        sl = round(entry_price + pip_step * 1.5, 5)

    # === FVG Count (basic placeholder) ===
    fvg_count = detect_fvg(df)

    return {
        "signal": signal,
        "reason": reasons,
        "entry": round(entry_price, 5),
        "tp_levels": tp_levels,
        "sl": sl,
        "rsi": float(round(rsi_now, 2)),
        "trend": trend,
        "fvg_count": fvg_count
    }

def detect_fvg(df: pd.DataFrame) -> int:
    count = 0
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i - 2]:
            count += 1
    return count