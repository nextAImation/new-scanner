# modules/liquidity_engine.py
import pandas as pd
from typing import Dict

def compute_liquidity_heatmap(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute liquidity metrics including volume analysis.
    Always returns dict with same keys.
    """
    default_result = {
        "liquidity": 0.0,
        "volume_score": 0.0, 
        "dollar_volume_avg": 0.0
    }
    
    try:
        if df is None or df.empty:
            return default_result
        
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return default_result
        
        if len(df) < 10:
            return default_result
        
        # Clean data
        clean_df = df[['close', 'volume']].dropna()
        if len(clean_df) < 10:
            return default_result
        
        # Calculate dollar volume
        dollar_volume = clean_df['close'] * clean_df['volume']
        avg_dollar_volume = float(dollar_volume.tail(20).mean())
        
        # Volume score based on recent volume vs average
        recent_volume = clean_df['volume'].tail(5).mean()
        avg_volume = clean_df['volume'].tail(20).mean()
        
        if avg_volume > 0:
            volume_score = min(max(float(recent_volume / avg_volume), 0.0), 2.0) / 2.0
        else:
            volume_score = 0.0
        
        # Overall liquidity score
        liquidity_score = min(max(float(volume_score * 0.7 + (min(avg_dollar_volume / 1000000, 1.0) * 0.3)), 0.0), 1.0)
        
        return {
            "liquidity": liquidity_score,
            "volume_score": volume_score,
            "dollar_volume_avg": avg_dollar_volume
        }
        
    except Exception:
        return default_result

if __name__ == "__main__":
    # Test the function
    test_df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1200, 800, 1500, 900, 1100, 1300, 700, 1600, 1000]
    })
    print(f"Liquidity heatmap: {compute_liquidity_heatmap(test_df)}")