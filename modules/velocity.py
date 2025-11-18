# modules/velocity.py
import pandas as pd
from typing import Union

def compute_velocity_signal(df: pd.DataFrame) -> float:
    """
    Compute velocity signal based on price momentum.
    Returns float between 0-1.
    """
    try:
        if df is None or df.empty or 'close' not in df.columns:
            return 0.0
        
        if len(df) < 5:
            return 0.0
        
        # Calculate momentum using recent price changes
        close_prices = df['close'].dropna()
        if len(close_prices) < 5:
            return 0.0
        
        # Use 3-period and 5-period momentum
        momentum_3 = (close_prices.iloc[-1] - close_prices.iloc[-4]) / close_prices.iloc[-4]
        momentum_5 = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
        
        # Combine and normalize to 0-1 range
        combined_momentum = (momentum_3 + momentum_5) / 2
        velocity = min(max(float(combined_momentum * 10), 0.0), 1.0)
        
        return velocity
        
    except Exception:
        return 0.0

if __name__ == "__main__":
    # Test the function
    test_df = pd.DataFrame({
        'close': [100, 102, 105, 103, 107, 110, 108, 112, 115, 113]
    })
    print(f"Velocity signal: {compute_velocity_signal(test_df)}")