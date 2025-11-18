# modules/compression.py
import pandas as pd
from typing import Union

def compute_compression_signal(df: pd.DataFrame) -> float:
    """
    Compute price compression signal indicating potential breakout.
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
        
        # Calculate volatility compression
        volatility_short = close_prices.tail(10).std()
        volatility_long = close_prices.tail(20).std()
        
        # Calculate Bollinger Band width (simplified)
        rolling_mean = close_prices.rolling(window=20).mean()
        rolling_std = close_prices.rolling(window=20).std()
        
        if len(rolling_std) > 0 and not pd.isna(rolling_std.iloc[-1]):
            bb_width = (rolling_std.iloc[-1] * 2) / rolling_mean.iloc[-1] if rolling_mean.iloc[-1] > 0 else 0
        else:
            bb_width = 0
        
        # Compression score (low volatility + narrowing BB)
        if volatility_long > 0:
            vol_compression = 1.0 - min(max(float(volatility_short / volatility_long), 0.0), 1.0)
        else:
            vol_compression = 0.0
        
        bb_compression = min(max(float(1.0 - bb_width * 10), 0.0), 1.0)
        
        compression_score = (vol_compression + bb_compression) / 2.0
        
        return compression_score
        
    except Exception:
        return 0.0

if __name__ == "__main__":
    # Test the function
    test_df = pd.DataFrame({
        'close': [100, 100.5, 99.8, 100.2, 100.1, 100.3, 99.9, 100.4, 100.2, 100.1]  # Low volatility
    })
    print(f"Compression signal: {compute_compression_signal(test_df)}")