# modules/risk_structure.py

import numpy as np
import pandas as pd

def risk_structure_score(df, lookback=50):
    """
    Risk-Structure Score (0–100)
    ترکیبی از:
        - volatility stability (ATR)
        - trend consistency (EMA alignment)
        - candle quality (upper/lower wicks)
        - pullback depth risk

    خروجی:
        dict {
            "score": float,
            "volatility": float,
            "trend_consistency": float,
            "wick_quality": float,
            "pullback_risk": float,
        }
    """

    # =======================
    # 0) ولیدیشن دیتافریم
    # =======================
    if df is None or len(df) < lookback + 5:
        return {
            "score": 0,
            "volatility": 0,
            "trend_consistency": 0,
            "wick_quality": 0,
            "pullback_risk": 0,
        }

    df = df.copy()
    df = df.tail(lookback)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # =======================
    # 1) ATR Volatility Stability (0–100)
    # =======================
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    atr_rel = (atr / close).iloc[-1]

    if np.isnan(atr_rel):
        vol_score = 0
    else:
        # ATR پایین → ریسک کمتر → امتیاز بیشتر
        vol_score = max(0, 100 - (atr_rel * 5000))
        vol_score = float(np.clip(vol_score, 0, 100))

    # =======================
    # 2) Trend Consistency EMA(20/50)
    # =======================
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    trend_consistency = 0
    if ema20.iloc[-1] > ema50.iloc[-1]:
        trend_consistency = 60 + min(40, abs(ema20.iloc[-1] - ema50.iloc[-1]) / close.iloc[-1] * 20000)
    else:
        trend_consistency = 20 - min(20, abs(ema20.iloc[-1] - ema50.iloc[-1]) / close.iloc[-1] * 20000)

    trend_consistency = float(np.clip(trend_consistency, 0, 100))

    # =======================
    # 3) Wick Quality (0–100)
    # =======================
    body = (close - df["open"]).abs()
    upper_wick = high - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - low

    avg_upper = (upper_wick / close).mean()
    avg_lower = (lower_wick / close).mean()

    wick_quality = max(0, 100 - (avg_upper + avg_lower) * 8000)
    wick_quality = float(np.clip(wick_quality, 0, 100))

    # =======================
    # 4) Pullback Depth Risk (0–100)
    # =======================
    recent_high = high.tail(10).max()
    pullback = (recent_high - close.iloc[-1]) / recent_high

    pullback_risk = float(np.clip(100 - pullback * 400, 0, 100))

    # =======================
    # 5) Final Composite Score
    # =======================
    final_score = (
        vol_score * 0.30 +
        trend_consistency * 0.35 +
        wick_quality * 0.20 +
        pullback_risk * 0.15
    )

    return {
        "score": round(float(final_score), 2),
        "volatility": round(vol_score, 2),
        "trend_consistency": round(trend_consistency, 2),
        "wick_quality": round(wick_quality, 2),
        "pullback_risk": round(pullback_risk, 2),
    }
