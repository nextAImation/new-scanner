# modules/trend_quality.py
import pandas as pd
from typing import Union

def trend_quality_score(df: pd.DataFrame) -> float:
    """
    Evaluate the quality and strength of the trend.
    Returns float between 0-1.
    """
    try:
        if df is None or df.empty or 'close' not in df.columns:
            return 0.0
        
        if len(df) < 20:
            return 0.0
        
        close_prices = df['close'].dropna()
        if len(close_prices) < 20:
            return 0.0
        
        # Calculate EMAs for trend detection
        ema_short = close_prices.ewm(span=10, adjust=False).mean()
        ema_long = close_prices.ewm(span=20, adjust=False).mean()
        
        # Trend strength based on EMA alignment and slope
        ema_diff = (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
        ema_slope = (ema_short.iloc[-1] - ema_short.iloc[-5]) / ema_short.iloc[-5]
        
        # Combine factors for trend quality score
        trend_strength = abs(float(ema_diff))
        trend_consistency = abs(float(ema_slope))
        
        quality_score = min(max((trend_strength + trend_consistency) * 2, 0.0), 1.0)
        
        return quality_score
        
    except Exception:
        return 0.0

if __name__ == "__main__":
    # Test the function
    test_df = pd.DataFrame({
        'close': list(range(100, 150))  # Strong uptrend
    })
    print(f"Trend quality: {trend_quality_score(test_df)}")