# modules/market_phase.py
import pandas as pd


def detect_market_phase(df: pd.DataFrame) -> str:
    """
    Detect current market phase: bull, bear, or sideways.
    Returns string: 'bull', 'bear', or 'sideways'
    """
    try:
        if df is None or df.empty or 'close' not in df.columns:
            return "unknown"
        
        if len(df) < 30:
            return "unknown"
        
        close_prices = df['close'].dropna()
        if len(close_prices) < 30:
            return "unknown"
        
        # Calculate moving averages
        sma_20 = close_prices.tail(20).mean()
        sma_30 = close_prices.tail(30).mean()
        current_price = close_prices.iloc[-1]
        
        # Calculate price change over different periods
        change_5 = (current_price - close_prices.iloc[-5]) / close_prices.iloc[-5] * 100
        change_10 = (current_price - close_prices.iloc[-10]) / close_prices.iloc[-10] * 100
        
        # Determine market phase
        if (current_price > sma_20 > sma_30 and 
            change_5 > 1.0 and change_10 > 2.0):
            return "bull"
        elif (current_price < sma_20 < sma_30 and 
              change_5 < -1.0 and change_10 < -2.0):
            return "bear"
        else:
            return "sideways"
            
    except Exception:
        return "unknown"

if __name__ == "__main__":
    # Test the function
    test_df = pd.DataFrame({
        'close': list(range(100, 130))  # Uptrend
    })
    print(f"Market phase: {detect_market_phase(test_df)}")