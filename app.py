from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

app = Flask(__name__)
CORS(app, origins="*", methods=['GET', 'POST', 'OPTIONS'])

TWELVEDATA_API_KEY = "1ea8171d30cc47c0880a12d2a067cb99"

# Twelve Data timeframe mapping
timeframe_map = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1day"
}

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Test Twelve Data connection with a simple quote request
        test_url = f"https://api.twelvedata.com/quote?symbol=EURUSD&apikey={TWELVEDATA_API_KEY}"
        resp = requests.get(test_url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            # Check if we got valid data (not an error response)
            if 'symbol' in data and not data.get('status') == 'error':
                twelvedata_status = "connected"
            else:
                twelvedata_status = "disconnected"
        else:
            twelvedata_status = "disconnected"

        return jsonify({
            "status": "healthy",
            "data_provider": "TwelveData",
            "twelvedata_status": twelvedata_status,
            "timestamp": datetime.now().isoformat(),
            "server_port": 5501
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "data_provider": "TwelveData",
            "twelvedata_status": "disconnected",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/get_signals', methods=['GET'])
def get_signals():
    try:
        symbol = request.args.get("symbol", "EURUSD")
        tf = request.args.get("tf", "M15").upper()
        bars = int(request.args.get("bars", 100))

        print(f"Received request: symbol={symbol}, tf={tf}, bars={bars}")

        if tf not in timeframe_map:
            return jsonify({"status": "error", "message": f"Invalid timeframe: {tf}"}), 400

        if bars < 20 or bars > 1000:
            return jsonify({"status": "error", "message": "Bars must be between 20 and 1000"}), 400

        interval = timeframe_map[tf]
        
        print(f"Timeframe interval: {interval}")
        print(f"Requested bars: {bars}")

        # Format symbol for Twelve Data (remove slash if present)
        if '/' in symbol:
            symbol_pair = symbol
        else:
            symbol_pair = f"{symbol[:3]}/{symbol[3:]}"

        # Twelve Data time series endpoint
        url = f"https://api.twelvedata.com/time_series?symbol={symbol_pair}&interval={interval}&outputsize={bars}&apikey={TWELVEDATA_API_KEY}"
        print(f"Fetching data from: {url}")

        try:
            response = requests.get(url, timeout=30)
            data = response.json()
        except ValueError:
            print("⚠️ JSON decoding error from Twelve Data.")
            return jsonify({"status": "error", "message": "Invalid JSON from Twelve Data"}), 500

        print(f"DEBUG: raw Twelve Data response: {data}")

        # Check for API errors
        if data.get("status") == "error":
            error_msg = data.get("message", "Unknown error from Twelve Data")
            print(f"⚠️ Twelve Data returned error: {error_msg}")
            return jsonify({"status": "error", "message": f"Twelve Data API error: {error_msg}"}), 400

        # Check if we have the expected data structure
        if 'values' not in data:
            print(f"⚠️ Missing 'values' key in Twelve Data response: keys present = {list(data.keys())}")
            return jsonify({"status": "error", "message": "Twelve Data response missing 'values' key"}), 500

        values = data.get('values', [])
        if not values or len(values) == 0:
            print("⚠️ No values in Twelve Data response.")
            return jsonify({"status": "error", "message": "No candle data returned by Twelve Data"}), 500

        if len(values) < 20:
            print(f"⚠️ Too few bars returned: count={len(values)}")
            return jsonify({"status": "error", "message": "Insufficient candle data received"}), 500

        # Convert Twelve Data format to DataFrame
        # Twelve Data returns data in reverse chronological order (newest first)
        df_data = []
        for candle in reversed(values):  # Reverse to get chronological order
            try:
                df_data.append({
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle.get("volume", 0))  # Some forex pairs might not have volume
                })
            except (ValueError, KeyError) as e:
                print(f"⚠️ Error parsing candle data: {e}")
                continue

        if not df_data:
            print("⚠️ No valid candle data after parsing.")
            return jsonify({"status": "error", "message": "No valid candle data after parsing"}), 500

        df = pd.DataFrame(df_data)

        if df.empty:
            print("⚠️ DataFrame from Twelve Data is empty.")
            return jsonify({"status": "error", "message": "Empty dataframe from Twelve Data"}), 500

        if bars < 200:
            print("⚠️ Warning: fewer than 200 bars requested, long-term MAs may be unreliable.")

        result = generate_signal(df, symbol)
        result["symbol"] = symbol
        result["timeframe"] = tf
        result["bars_requested"] = bars
        result["bars_received"] = len(df)
        result["timestamp"] = datetime.now().isoformat()

        return jsonify(result)

    except Exception as e:
        print(f"Unexpected server error: {str(e)}")
        return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"}), 500

def generate_signal(df, symbol=""):
    if df.empty or len(df) < 50:
        return {
            "status": "error",
            "signal": "NEUTRAL",
            "reason": ["Not enough data for analysis"],
            "entry": None,
            "tp_levels": [],
            "sl": None,
            "rsi": None,
            "sma10": None,
            "sma20": None,
            "trend": "unknown",
            "confidence": "low"
        }
    try:
        df['diff'] = df['close'].diff()
        gain = df['diff'].clip(lower=0)
        loss = -df['diff'].clip(upper=0)
        avg_gain = gain.ewm(span=14).mean()
        avg_loss = loss.ewm(span=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        rsi_now = df['rsi'].iloc[-1]

        df['sma10'] = df['close'].rolling(10).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()

        sma10_now = df['sma10'].iloc[-1]
        sma20_now = df['sma20'].iloc[-1]

        trend = "unknown"
        reasons = []

        if len(df) >= 200 and not pd.isna(df['ma50'].iloc[-1]) and not pd.isna(df['ma200'].iloc[-1]):
            if df['ma50'].iloc[-1] > df['ma200'].iloc[-1]:
                trend = "UP"
                reasons.append("MA50 > MA200 (Uptrend)")
            elif df['ma50'].iloc[-1] < df['ma200'].iloc[-1]:
                trend = "DOWN"
                reasons.append("MA50 < MA200 (Downtrend)")

        signal = "NEUTRAL"
        entry = df['close'].iloc[-1]
        pip = 0.01 if "JPY" in symbol else 0.0001
        pip_step = pip * 20

        if rsi_now < 30 and trend == "UP":
            signal = "BUY"
            reasons.append(f"RSI oversold ({rsi_now:.1f}) + Uptrend")
        elif rsi_now > 70 and trend == "DOWN":
            signal = "SELL"
            reasons.append(f"RSI overbought ({rsi_now:.1f}) + Downtrend")
        elif not pd.isna(sma10_now) and not pd.isna(sma20_now):
            if sma10_now > sma20_now and rsi_now < 50:
                signal = "BUY"
                reasons.append("SMA10 > SMA20 + RSI < 50")
            elif sma10_now < sma20_now and rsi_now > 50:
                signal = "SELL"
                reasons.append("SMA10 < SMA20 + RSI > 50")

        if not reasons:
            reasons.append("No strong signal detected")

        if signal == "BUY":
            tp = [round(entry + pip_step * i, 5) for i in range(1, 4)]
            sl = round(entry - pip_step * 1.5, 5)
        elif signal == "SELL":
            tp = [round(entry - pip_step * i, 5) for i in range(1, 4)]
            sl = round(entry + pip_step * 1.5, 5)
        else:
            tp = []
            sl = None

        return {
            "status": "success",
            "signal": signal,
            "reason": reasons,
            "entry": round(entry, 5),
            "tp_levels": tp,
            "sl": sl,
            "rsi": round(float(rsi_now), 2) if not pd.isna(rsi_now) else None,
            "sma10": round(float(sma10_now), 5) if not pd.isna(sma10_now) else None,
            "sma20": round(float(sma20_now), 5) if not pd.isna(sma20_now) else None,
            "trend": trend,
            "confidence": "medium" if signal != "NEUTRAL" else "low"
        }
    except Exception as e:
        print(f"Error in generate_signal: {str(e)}")
        return {
            "status": "error",
            "signal": "NEUTRAL",
            "reason": [f"Analysis error: {str(e)}"],
            "entry": None,
            "tp_levels": [],
            "sl": None,
            "rsi": None,
            "sma10": None,
            "sma20": None,
            "trend": "unknown",
            "confidence": "low"
        }

@app.route('/', methods=['GET'])
def home():
    return "✅ Trading Signals API is running. Use /get_signals or /health."


if __name__ == '__main__':
    print("Starting Trading Signals Server...")
    print("Make sure to install required packages: pip install flask flask-cors requests pandas numpy")
    import os
    port = int(os.environ.get("PORT", 5501))
    app.run(debug=False, host='0.0.0.0', port=port)
