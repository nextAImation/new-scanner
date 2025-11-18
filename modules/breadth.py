# breadth.py
import pandas as pd
import numpy as np


def compute_breadth_summary(df: pd.DataFrame, lookback: int = 100) -> dict:
    """محاسبه یک خلاصه ساده از پهنای حرکت (Breadth) روی خود نماد.

    این تابع قرار نیست مثل Breadth واقعی بازار (تعداد نمادهای مثبت/منفی) کار کند،
    فقط روی همین df یک تصویر از میزان «گستره‌ی کندل‌های صعودی/نزولی» می‌دهد
    تا در ریپورت اسکنر استفاده شود.

    ورودی:
        df: دیتافریم کندل‌ها با ستون‌های حداقل: open, close, high, low
        lookback: تعداد کندل آخر برای محاسبه آمار

    خروجی:
        دیکشنری JSON-دوست برای قرار گرفتن مستقیم در گزارش:
        {
            "sample_size": n,
            "up_ratio": ...,
            "down_ratio": ...,
            "flat_ratio": ...,
            "avg_range": ...,
            "avg_body": ...,
            "strong_trend_ratio": ...,
        }
    """
    required_cols = {"open", "close", "high", "low"}
    if not required_cols.issubset(df.columns):
        return {
            "sample_size": 0,
            "up_ratio": None,
            "down_ratio": None,
            "flat_ratio": None,
            "avg_range": None,
            "avg_body": None,
            "strong_trend_ratio": None,
        }

    if len(df) == 0:
        return {
            "sample_size": 0,
            "up_ratio": None,
            "down_ratio": None,
            "flat_ratio": None,
            "avg_range": None,
            "avg_body": None,
            "strong_trend_ratio": None,
        }

    # فقط n کندل آخر
    df = df.tail(lookback).copy()
    n = len(df)
    if n == 0:
        return {
            "sample_size": 0,
            "up_ratio": None,
            "down_ratio": None,
            "flat_ratio": None,
            "avg_range": None,
            "avg_body": None,
            "strong_trend_ratio": None,
        }

    opens = df["open"].astype(float)
    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)

    # دسته‌بندی کندل‌ها
    up_mask = closes > opens
    down_mask = closes < opens
    flat_mask = closes == opens

    up_cnt = int(up_mask.sum())
    down_cnt = int(down_mask.sum())
    flat_cnt = int(flat_mask.sum())

    up_ratio = up_cnt / n
    down_ratio = down_cnt / n
    flat_ratio = flat_cnt / n

    # برد کندل و بدنه
    ranges = (highs - lows).replace([np.inf, -np.inf], np.nan).dropna()
    bodies = (closes - opens).abs().replace([np.inf, -np.inf], np.nan).dropna()

    avg_range = float(ranges.mean()) if not ranges.empty else None
    avg_body = float(bodies.mean()) if not bodies.empty else None

    # کندل‌هایی که بدنه‌شان حداقل ۷۵٪ بردشان است → روند قوی‌تر
    strong_trend_mask = (bodies >= 0.75 * ranges) & (ranges > 0)
    strong_trend_ratio = float(strong_trend_mask.sum() / n)

    return {
        "sample_size": int(n),
        "up_ratio": float(up_ratio),
        "down_ratio": float(down_ratio),
        "flat_ratio": float(flat_ratio),
        "avg_range": avg_range,
        "avg_body": avg_body,
        "strong_trend_ratio": strong_trend_ratio,
    }
