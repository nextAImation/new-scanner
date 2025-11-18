# modules/sr_reliability.py
import pandas as pd
from typing import Union

def compute_sr_reliability(df: pd.DataFrame) -> float:
    """
    Compute support/resistance reliability score.
    Returns float between 0-1.
    """
    try:
        if df is None or df.empty:
            return 0.0
        
        required_cols = ['high', 'low']
        if not all(col in df.columns for col in required_cols):
            return 0.0
        
        if len(df) < 20:
            return 0.0
        
        # Clean data
        clean_df = df[['high', 'low']].dropna()
        if len(clean_df) < 20:
            return 0.0
        
        # Calculate price consolidation ratio
        recent_high = clean_df['high'].tail(10).max()
        recent_low = clean_df['low'].tail(10).min()
        price_range = recent_high - recent_low
        
        avg_true_range = (clean_df['high'] - clean_df['low']).tail(10).mean()
        
        if price_range > 0:
            # Lower consolidation = higher S/R reliability
            consolidation_ratio = avg_true_range / price_range
            reliability = min(max(float(1.0 - consolidation_ratio), 0.0), 1.0)
        else:
            reliability = 0.0
        
        return reliability
        
    except Exception:
        return 0.0

if __name__ == "__main__":
    # Test the function
    test_df = pd.DataFrame({
        'high': [105, 106, 104, 107, 105, 106, 104, 107, 105, 106],
        'low': [95, 96, 94, 97, 95, 96, 94, 97, 95, 96]
    })
    print(f"SR reliability: {compute_sr_reliability(test_df)}")