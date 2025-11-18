# modules/dollar_volume.py
import pandas as pd
from typing import Union

def compute_dollar_volume(df: pd.DataFrame) -> float:
    """
    Compute average dollar volume (price * volume).
    Returns float value.
    """
    try:
        if df is None or df.empty:
            return 0.0
        
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return 0.0
        
        if len(df) < 5:
            return 0.0
        
        # Clean data
        clean_df = df[['close', 'volume']].dropna()
        if len(clean_df) < 5:
            return 0.0
        
        # Calculate dollar volume and return 20-period average
        dollar_volume = clean_df['close'] * clean_df['volume']
        avg_dollar_volume = float(dollar_volume.tail(20).mean())
        
        return avg_dollar_volume
        
    except Exception:
        return 0.0

if __name__ == "__main__":
    # Test the function
    test_df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105],
        'volume': [1000, 1200, 800, 1500, 900, 1100]
    })
    print(f"Dollar volume: {compute_dollar_volume(test_df)}")