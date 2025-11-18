# quick_scan.py — Unified Intelligence Edition (Streamlit Compatible)

import argparse, os, json, datetime as dt, time
from typing import Dict, List, Tuple, Optional, Any
import requests, ccxt, pandas as pd, numpy as np, yfinance as yf
from dotenv import load_dotenv
load_dotenv()

# === IMPORT NEW MODULES ===
try:
    from modules.velocity import compute_velocity_signal
    from modules.trend_quality import trend_quality_score
    from modules.liquidity_engine import compute_liquidity_heatmap
    from modules.market_phase import detect_market_phase
    from modules.risk_structure import risk_structure_score
    from modules.sr_reliability import compute_sr_reliability
    from modules.breadth import compute_breadth_summary
    from modules.compression import compute_compression_signal
    from modules.dollar_volume import compute_dollar_volume
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some modules not available: {e}")
    MODULES_AVAILABLE = False

# === AUTO CSV GENERATOR (HEATMAP) ===
def generate_market_csv(csv_path: str = "market_heatmap.csv") -> Optional[str]:
    """
    Auto-generate a fresh market heatmap CSV from CoinGecko.
    Contains: symbol, price, change_24h, volume_24h, market_cap
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        rows = []
        for d in data:
            rows.append(
                {
                    "symbol": d.get("symbol", "").upper(),
                    "price": d.get("current_price"),
                    "change_24h": d.get("price_change_percentage_24h"),
                    "volume_24h": d.get("total_volume"),
                    "market_cap": d.get("market_cap"),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        print(f"✅ Auto-generated heatmap CSV: {csv_path}")
        return csv_path

    except Exception as e:
        print(f"❌ Failed to generate heatmap CSV: {e}")
        return None

# === AUTO MARKET ARCHIVE (CSV + JSON) ===
def generate_market_archive(
    heatmap_csv: str = "market_heatmap.csv",
    archive_csv: str = "market_archive.csv",
    archive_json: str = "market_archive.json",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Build a richer market archive from the heatmap CSV.
    Produces:
      - archive_csv: flattened summary (GLOBAL / DOMINANCE / TOP_GAINERS / TOP_LOSERS / TOP_VOLUME)
      - archive_json: structured JSON for advanced GPT analysis
    """
    try:
        # Ensure heatmap exists
        if not os.path.exists(heatmap_csv):
            print(f"ℹ Heatmap CSV {heatmap_csv} not found → regenerating")
            generate_market_csv(heatmap_csv)

        if not os.path.exists(heatmap_csv):
            print(f"❌ Still no heatmap CSV at {heatmap_csv} → abort market archive")
            return None, None

        df = pd.read_csv(heatmap_csv)

        # Basic sanity
        if df.empty:
            print("❌ Heatmap CSV is empty → abort market archive")
            return None, None

        # Ensure columns
        for col in ["symbol", "price", "change_24h", "volume_24h", "market_cap"]:
            if col not in df.columns:
                df[col] = None

        # GLOBAL STATS (approx)
        mcaps = df["market_cap"].fillna(0)
        total_mcap = float(mcaps.sum()) if mcaps.sum() > 0 else None
        sample_size = int(len(df))

        # DOMINANCE (approx from subset)
        btc_row = df[df["symbol"] == "BTC"]
        eth_row = df[df["symbol"] == "ETH"]

        def _safe_cap(row):
            if row.empty:
                return 0.0
            v = row["market_cap"].iloc[0]
            try:
                return float(v) if pd.notna(v) else 0.0
            except Exception:
                return 0.0

        btc_cap = _safe_cap(btc_row)
        eth_cap = _safe_cap(eth_row)

        btc_dom = eth_dom = others_dom = None
        if total_mcap and total_mcap > 0:
            btc_dom = btc_cap / total_mcap * 100.0 if btc_cap > 0 else 0.0
            eth_dom = eth_cap / total_mcap * 100.0 if eth_cap > 0 else 0.0
            others_dom = max(0.0, 100.0 - btc_dom - eth_dom)

        # TOP LISTS
        df_change = df.copy()
        df_change = df_change[pd.notna(df_change["change_24h"])]
        df_change_pos = df_change.sort_values("change_24h", ascending=False)
        df_change_neg = df_change.sort_values("change_24h", ascending=True)

        top_gainers = df_change_pos.head(5)
        top_losers = df_change_neg.head(5)

        df_vol = df.copy()
        df_vol = df_vol[pd.notna(df_vol["volume_24h"])]
        top_volume = df_vol.sort_values("volume_24h", ascending=False).head(5)

        # JSON ARCHIVE
        archive = {
            "generated_at": dt.datetime.utcnow().isoformat(),
            "global": {
                "approx_total_market_cap_usd": total_mcap,
                "sample_size": sample_size,
            },
            "dominance": {
                "BTC": btc_dom,
                "ETH": eth_dom,
                "OTHERS": others_dom,
            },
            "top_gainers": [
                {
                    "symbol": str(r["symbol"]),
                    "price": float(r["price"]) if pd.notna(r["price"]) else None,
                    "change_24h": float(r["change_24h"])
                    if pd.notna(r["change_24h"])
                    else None,
                    "volume_24h": float(r["volume_24h"])
                    if pd.notna(r["volume_24h"])
                    else None,
                }
                for _, r in top_gainers.iterrows()
            ],
            "top_losers": [
                {
                    "symbol": str(r["symbol"]),
                    "price": float(r["price"]) if pd.notna(r["price"]) else None,
                    "change_24h": float(r["change_24h"])
                    if pd.notna(r["change_24h"])
                    else None,
                    "volume_24h": float(r["volume_24h"])
                    if pd.notna(r["volume_24h"])
                    else None,
                }
                for _, r in top_losers.iterrows()
            ],
            "top_volume": [
                {
                    "symbol": str(r["symbol"]),
                    "price": float(r["price"]) if pd.notna(r["price"]) else None,
                    "change_24h": float(r["change_24h"])
                    if pd.notna(r["change_24h"])
                    else None,
                    "volume_24h": float(r["volume_24h"])
                    if pd.notna(r["volume_24h"])
                    else None,
                }
                for _, r in top_volume.iterrows()
            ],
        }

        with open(archive_json, "w", encoding="utf-8") as f:
            json.dump(archive, f, ensure_ascii=False, indent=2)

        # CSV ARCHIVE (flattened)
        rows_csv = []

        # GLOBAL
        rows_csv.append(
            {
                "section": "GLOBAL",
                "name": "approx_total_market_cap_usd",
                "value": total_mcap,
                "extra": "",
            }
        )
        rows_csv.append(
            {
                "section": "GLOBAL",
                "name": "sample_size",
                "value": sample_size,
                "extra": "",
            }
        )

        # DOMINANCE
        rows_csv.append(
            {"section": "DOMINANCE", "name": "BTC", "value": btc_dom, "extra": "%"}
        )
        rows_csv.append(
            {"section": "DOMINANCE", "name": "ETH", "value": eth_dom, "extra": "%"}
        )
        rows_csv.append(
            {
                "section": "DOMINANCE",
                "name": "OTHERS",
                "value": others_dom,
                "extra": "%",
            }
        )

        # TOP LIST HELPERS
        def _rows_from_df(section: str, frame: pd.DataFrame) -> List[dict]:
            out_rows = []
            for _, r in frame.iterrows():
                sym = str(r["symbol"])
                chg = r["change_24h"]
                price = r["price"]
                vol = r["volume_24h"]
                extra = f"price={price}, vol={vol}"
                out_rows.append(
                    {
                        "section": section,
                        "name": sym,
                        "value": chg,
                        "extra": extra,
                    }
                )
            return out_rows

        rows_csv.extend(_rows_from_df("TOP_GAINERS", top_gainers))
        rows_csv.extend(_rows_from_df("TOP_LOSERS", top_losers))
        rows_csv.extend(_rows_from_df("TOP_VOLUME", top_volume))

        df_arch = pd.DataFrame(rows_csv, columns=["section", "name", "value", "extra"])
        df_arch.to_csv(archive_csv, index=False)

        print(f"✅ Market archive CSV:  {archive_csv}")
        print(f"✅ Market archive JSON: {archive_json}")
        return archive_csv, archive_json

    except Exception as e:
        print(f"❌ Failed to generate market archive: {e}")
        return None, None

# ================== BASIC INDICATORS ==================
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(s, n=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.rolling(n).mean() / (dn.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()

def fix_ohlc_spikes(df):
    if df.empty or len(df) < 2:
        return df

    df = df.copy()

    flat = df["high"] == df["low"]
    df.loc[flat, "high"] = df["close"] * 1.0005
    df.loc[flat, "low"] = df["close"] * 0.9995

    zero_vol = df["volume"] == 0
    df.loc[zero_vol, "high"] = df["close"] * 1.0007
    df.loc[zero_vol, "low"] = df["close"] * 0.9993

    return df

def adx(df, n=14):
    if df is None or len(df) < n + 5:
        return 0.0

    df = fix_ohlc_spikes(df).copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm_raw = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm_raw, index=df.index)
    minus_dm = pd.Series(minus_dm_raw, index=df.index)

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_val = rma(tr, n)
    atr_val = (
        atr_val.replace([0, np.inf, -np.inf], np.nan)
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    plus_di = 100 * rma(plus_dm, n) / (atr_val + 1e-12)
    minus_di = 100 * rma(minus_dm, n) / (atr_val + 1e-12)

    plus_di = plus_di.replace([np.inf, -np.inf], np.nan).fillna(0)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan).fillna(0)

    denominator = plus_di + minus_di
    dx = np.where(denominator > 0, abs(plus_di - minus_di) / denominator * 100, 0)
    dx = pd.Series(dx, index=df.index)

    adx_series = rma(dx, n)
    valid = adx_series.dropna()

    return float(valid.iloc[-1]) if len(valid) else 0.0

def atr(df, n=14):
    if len(df) < 2:
        return None
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()

def vbp_poc(df, bins=24):
    if len(df) < 10:
        return None, None, None
    px = (df["high"] + df["low"] + df["close"]) / 3
    hist, edges = np.histogram(px, bins=bins, weights=df["volume"])
    i = np.argmax(hist)
    return (edges[i] + edges[i + 1]) / 2, edges[i], edges[i + 1]

# ================== FVG DETECTOR ==================
def detect_fvg(df):
    """
    Fair Value Gap (FVG) Detector
    شناسایی شکاف‌های قیمتی در بازار
    """
    if len(df) < 3:
        return []

    fvg_list = []
    for i in range(2, len(df)):
        try:
            if df["low"].iloc[i] > df["high"].iloc[i - 2]:
                fvg_list.append(
                    {
                        "type": "bullish",
                        "start": float(df["high"].iloc[i - 2]),
                        "end": float(df["low"].iloc[i]),
                        "time": df.index[i].strftime("%Y-%m-%d"),
                    }
                )
            if df["high"].iloc[i] < df["low"].iloc[i - 2]:
                fvg_list.append(
                    {
                        "type": "bearish",
                        "start": float(df["low"].iloc[i - 2]),
                        "end": float(df["high"].iloc[i]),
                        "time": df.index[i].strftime("%Y-%m-%d"),
                    }
                )
        except IndexError:
            continue

    return fvg_list[-3:] if len(fvg_list) > 0 else []

# ================== MARKET STRUCTURE ==================
def find_pivots(df, left=2, right=2):
    """
    بر اساس high/low، pivot بالا/پایین را مشخص می‌کند.
    left/right یعنی چند کندل دو طرف باید پایین‌تر/بالاتر باشند.
    """
    if len(df) < (left + right + 1):
        return df

    high = df["high"].values
    low = df["low"].values
    n = len(df)

    pivots_high = np.zeros(n, dtype=bool)
    pivots_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_h = high[i - left : i + right + 1]
        window_l = low[i - left : i + right + 1]
        if high[i] == window_h.max():
            pivots_high[i] = True
        if low[i] == window_l.min():
            pivots_low[i] = True

    df = df.copy()
    df["pivot_high"] = pivots_high
    df["pivot_low"] = pivots_low
    return df

def label_structure_from_pivots(df, lookback_swings=6):
    if len(df) < 4:
        return [], "unknown", False, False

    swings = []
    for idx, row in df.iterrows():
        if row.get("pivot_high"):
            swings.append({"type": "H", "price": row["high"], "time": idx, "idx": idx})
        elif row.get("pivot_low"):
            swings.append({"type": "L", "price": row["low"], "time": idx, "idx": idx})

    if len(swings) < 4:
        return swings, "unknown", False, False

    swings = swings[-lookback_swings:]
    last, prev, prev2 = swings[-1], swings[-2], swings[-3]

    if last["type"] == prev["type"] == "H":
        structure_tag = "HH" if last["price"] > prev["price"] else "LH"
    elif last["type"] == prev["type"] == "L":
        structure_tag = "HL" if last["price"] > prev["price"] else "LL"
    else:
        structure_tag = "unknown"

    bos = False
    if last["type"] == "H" and prev["type"] == "H" and last["price"] > prev["price"]:
        bos = True
    if last["type"] == "L" and prev["type"] == "L" and last["price"] < prev["price"]:
        bos = True

    choch = False
    if prev2["type"] == prev["type"] == "H" and last["type"] == "L":
        choch = True
    if prev2["type"] == prev["type"] == "L" and last["type"] == "H":
        choch = True

    return swings, structure_tag, choch, bos

# ================== ADVANCED STRUCTURE & LIQUIDITY ==================
def build_structure_zones_from_swings(swings, tolerance_pct=0.5, max_zones=6):
    """
    از swings (H/L) ناحیه‌های S/R ساده می‌سازد.
    """
    if not swings:
        return []

    highs = [s["price"] for s in swings if s["type"] == "H"]
    lows = [s["price"] for s in swings if s["type"] == "L"]

    def _cluster(levels, lvl_type):
        levels = sorted(levels)
        clusters = []
        for p in levels:
            if not clusters:
                clusters.append([p])
            else:
                last_cluster = clusters[-1]
                center = sum(last_cluster) / len(last_cluster)
                if abs(p - center) / center * 100 <= tolerance_pct:
                    last_cluster.append(p)
                else:
                    clusters.append([p])
        zones = []
        for cl in clusters:
            center = sum(cl) / len(cl)
            zones.append({"type": lvl_type, "level": float(center), "count": len(cl)})
        zones = sorted(zones, key=lambda z: z["count"], reverse=True)
        return zones[:max_zones]

    zones = []
    if highs:
        zones.extend(_cluster(highs, "resistance"))
    if lows:
        zones.extend(_cluster(lows, "support"))
    return zones

def compute_trendline_slope(series: pd.Series, window: int = 50):
    """
    شیب تقریبی خط روند روی آخرین window کندل بر اساس رگرسیون خطی.
    """
    if series is None or len(series) < 10:
        return None, "unknown"

    s = series.dropna().tail(window)
    if len(s) < 10:
        return None, "unknown"

    y = s.values
    x = np.arange(len(y))

    try:
        coef = np.polyfit(x, y, 1)
        slope = coef[0] / (y.mean() + 1e-9) * 100
    except Exception:
        return None, "unknown"

    if slope > 0.2:
        label = "up"
    elif slope < -0.2:
        label = "down"
    else:
        label = "flat"

    return float(slope), label

def liquidity_map_from_vpvr(df: pd.DataFrame, bins: int = 24, top_n: int = 3):
    """
    نقشه‌ی نقدینگی ساده بر اساس توزیع حجم روی قیمت (Volume Profile).
    """
    if df is None or len(df) < 20:
        return []

    px = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]
    hist, edges = np.histogram(px, bins=bins, weights=vol)
    total_vol = hist.sum()
    if total_vol <= 0:
        return []

    zones = []
    for i, v in enumerate(hist):
        if v <= 0:
            continue
        zones.append(
            {
                "price_low": float(edges[i]),
                "price_high": float(edges[i + 1]),
                "volume_share_pct": float(v / total_vol * 100.0),
            }
        )

    zones = sorted(zones, key=lambda z: z["volume_share_pct"], reverse=True)
    return zones[:top_n]

def detect_candle_patterns(df: pd.DataFrame, lookback: int = 5):
    """
    شناسایی چند الگوی کندلی ساده روی HTF.
    """
    patterns = []
    if df is None or len(df) < 3:
        return patterns

    data = df.tail(lookback + 1).copy()
    for i in range(1, len(data)):
        row_prev = data.iloc[i - 1]
        row = data.iloc[i]

        o1, c1, h1, l1, t1 = (
            row_prev["open"],
            row_prev["close"],
            row_prev["high"],
            row_prev["low"],
            row_prev.name,
        )
        o2, c2, h2, l2, t2 = (
            row["open"],
            row["close"],
            row["high"],
            row["low"],
            row.name,
        )

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range2 = h2 - l2
        upper_wick2 = h2 - max(o2, c2)
        lower_wick2 = min(o2, c2) - l2

        if body1 > 0 and body2 > 0:
            if c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1:
                patterns.append({"time": t2.isoformat(), "pattern": "bullish_engulfing"})
            if c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
                patterns.append({"time": t2.isoformat(), "pattern": "bearish_engulfing"})

        if range2 > 0:
            if lower_wick2 > body2 * 2 and upper_wick2 < body2:
                patterns.append({"time": t2.isoformat(), "pattern": "hammer"})
            if upper_wick2 > body2 * 2 and lower_wick2 < body2:
                patterns.append({"time": t2.isoformat(), "pattern": "shooting_star"})

        if range2 > 0 and body2 <= range2 * 0.1:
            patterns.append({"time": t2.isoformat(), "pattern": "doji"})

    return patterns[-lookback:]

# ================== CRASH RISK ENGINE ==================
def compute_crash_risk(
    price,
    adx,
    rsi,
    atr_rel,
    bb_width,
    ema_slope,
    volume_climax,
    structure_tag,
    choch,
    liquidity_map,
    fvg_list,
):
    """
    Computes a robust crash risk score (0-100).
    """
    score = 0
    reasons = []

    if rsi is not None and rsi < 28:
        score += 15
        reasons.append("RSI < 28 (oversold breakdown)")

    if adx is not None and adx < 15:
        score += 10
        reasons.append("ADX < 15 (weak trend, vulnerable)")

    if atr_rel is not None and atr_rel > 0.045:
        score += 20
        reasons.append("ATR relative > 4.5% (volatility spike)")

    if bb_width is not None and bb_width > 0.04:
        score += 15
        reasons.append("BB width > 0.04 (compression → expansion)")

    if ema_slope is not None and ema_slope < -2.5:
        score += 15
        reasons.append("EMA20 slope < -2.5% (down momentum)")

    if volume_climax:
        score += 10
        reasons.append("Volume climax (distribution top)")

    if choch and structure_tag in ["LL", "LH"]:
        score += 10
        reasons.append("Bearish CHOCH + LL/LH")

    if liquidity_map and len(liquidity_map) > 0:
        biggest = liquidity_map[0]["volume_share_pct"]
        if biggest < 7:
            score += 10
            reasons.append("Weak liquidity nodes (vacuum)")

    if any(x["type"] == "bearish" for x in fvg_list):
        score += 10
        reasons.append("Bearish FVG detected")

    score = min(score, 100)
    flag = score >= 40

    return score, flag, reasons

# ================== VOLATILITY ENGINE ==================
def bollinger_bands(s, n=20, k=2.0):
    if len(s) < n:
        return None, None, None, None
    ma = s.rolling(n).mean()
    std = s.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    width = (upper - lower) / ma
    return ma, upper, lower, width

def volume_zscore(v, window=20):
    if len(v) < window:
        return pd.Series([0] * len(v), index=v.index)
    mean = v.rolling(window).mean()
    std = v.rolling(window).std()
    z = (v - mean) / (std + 1e-9)
    return z

def classify_volatility(bb_width, atr_rel, adx_value, ema_slope=None):
    """
    bb_width: آخرین مقدار BB Width
    atr_rel: ATR / price
    adx_value: ADX14
    ema_slope: شیب EMA20
    """
    if bb_width is None or atr_rel is None:
        return "unknown"

    if adx_value is not None and ema_slope is not None:
        if adx_value > 35 and abs(ema_slope) > 0.002:
            return "strong_trend"

    if bb_width < 0.01 and atr_rel < 0.01:
        return "calm"
    if bb_width < 0.015 and atr_rel < 0.015:
        return "squeeze"
    if bb_width > 0.03 or atr_rel > 0.03:
        return "high_vol"
    if 20 <= (adx_value or 0) <= 35:
        return "trending"
    return "normal"

# ================== CORRELATION & LIQUIDITY ==================
def corr_beta(df_target, df_btc, window=90):
    print("=== [corr_beta] ENTER ===")
    if df_target is None or df_btc is None:
        print("[corr_beta] df_target or df_btc is None → return (None, None)")
        print("=== [corr_beta] EXIT ===")
        return None, None

    if df_target.empty or df_btc.empty:
        print(
            f"[corr_beta] empty dfs → target_empty={df_target.empty}, btc_empty={df_btc.empty} → return (None, None)"
        )
        print("=== [corr_beta] EXIT ===")
        return None, None

    df_target = df_target.sort_index()
    df_target = df_target[~df_target.index.duplicated(keep="last")]
    df_btc = df_btc.sort_index()
    df_btc = df_btc[~df_btc.index.duplicated(keep="last")]

    print(
        f"[corr_beta] after sort & dedup | "
        f"len(target)={len(df_target)}, len(btc)={len(df_btc)}, "
        f"target_start={df_target.index[0]}, target_end={df_target.index[-1]}, "
        f"btc_start={df_btc.index[0]}, btc_end={df_btc.index[-1]}"
    )

    aligned = df_target[["close"]].join(
        df_btc[["close"]].rename(columns={"close": "btc_close"}),
        how="inner",
    )
    print(
        f"[corr_beta] aligned len={len(aligned)} | "
        f"aligned_start={aligned.index[0] if len(aligned) > 0 else 'NONE'}, "
        f"aligned_end={aligned.index[-1] if len(aligned) > 0 else 'NONE'}"
    )

    if len(aligned) < 30:
        print(
            f"[corr_beta] NOT ENOUGH aligned candles ({len(aligned)}) → return (None, None)"
        )
        print("=== [corr_beta] EXIT ===")
        return None, None

    ret = np.log(aligned["close"] / aligned["close"].shift(1)).dropna()
    ret_btc = np.log(aligned["btc_close"] / aligned["btc_close"].shift(1)).dropna()

    aligned_ret = pd.concat([ret, ret_btc], axis=1).dropna()
    aligned_ret.columns = ["r", "r_btc"]

    print(f"[corr_beta] aligned_ret len={len(aligned_ret)} | window={window}")

    if len(aligned_ret) < 30:
        print(
            f"[corr_beta] NOT ENOUGH return rows ({len(aligned_ret)}) → return (None, None)"
        )
        print("=== [corr_beta] EXIT ===")
        return None, None

    window_slice = aligned_ret.iloc[-min(window, len(aligned_ret)) :]
    corr = window_slice["r"].corr(window_slice["r_btc"])
    var_btc = window_slice["r_btc"].var()

    print(f"[corr_beta] pre-check | corr={corr}, var_btc={var_btc}")

    if pd.isna(corr) or var_btc <= 0:
        print(
            f"[corr_beta] INVALID corr or var_btc → corr={corr}, var_btc={var_btc} → return (None, None)"
        )
        print("=== [corr_beta] EXIT ===")
        return None, None

    cov = window_slice.cov().loc["r", "r_btc"]
    beta = cov / var_btc if var_btc > 0 else None

    print(
        f"[corr_beta] SUCCESS | corr={float(corr)}, beta={float(beta) if beta is not None else None}"
    )
    print("=== [corr_beta] EXIT ===")
    return float(corr), float(beta) if beta is not None else None

def dev_from_extremes(df, days=90):
    """
    بر اساس داده df (تایم‌فریم 1d یا معادل تقریبی)،
    فاصله درصدی از high/low دوره را می‌دهد.
    """
    if df.empty:
        return None, None

    df_clean = df.dropna(subset=["high", "low", "close"])
    if df_clean.empty:
        return None, None

    sub = df_clean.tail(days)
    if len(sub) < days * 0.8:
        return None, None

    hi = sub["high"].max()
    lo = sub["low"].min()
    last_price = sub["close"].iloc[-1]

    dev_hi = float((last_price - hi) / hi * 100.0) if hi > 0 else None
    dev_lo = float((last_price - lo) / lo * 100.0) if lo > 0 else None
    return dev_hi, dev_lo

def liquidity_score(df, days=30):
    """
    نقدشوندگی تقریبی: مجموع (price * volume) 30 روز آخر
    """
    if df.empty:
        return None

    df_clean = df.dropna(subset=["close", "volume"])
    if df_clean.empty:
        return None

    sub = df_clean.tail(days)
    notional = (sub["close"] * sub["volume"]).sum()
    return float(notional)

# ================== DATA QUALITY ==================
def assess_data_quality(df):
    """
    یک ارزیابی خیلی ساده ولی کاربردی
    """
    if df.empty:
        return {"status": "bad", "reason": "empty"}

    n = len(df)
    nan_o = df["open"].isna().sum()
    nan_h = df["high"].isna().sum()
    nan_l = df["low"].isna().sum()
    nan_c = df["close"].isna().sum()

    flat_bars = ((df["high"] == df["low"]) & (df["low"] == df["close"])).sum()
    zero_volume = (df["volume"] == 0).sum()

    issues = []
    if nan_o + nan_h + nan_l + nan_c > 0:
        issues.append("nan_ohlc")
    if flat_bars > 0:
        issues.append("flat_bars")
    if zero_volume > 0:
        issues.append("zero_volume")

    status = "good" if not issues else "warning"
    if (nan_o + nan_h + nan_l + nan_c) > n * 0.1:
        status = "bad"

    return {
        "status": status,
        "issues": issues,
        "nan_ohlc": int(nan_o + nan_h + nan_l + nan_c),
        "flat_bars": int(flat_bars),
        "zero_volume": int(zero_volume),
        "total_bars": int(n),
    }

# ================== FETCH ==================
def _ex(eid):
    return getattr(ccxt, eid)({"enableRateLimit": True, "timeout": 20000})

def fetch_ohlcv(symbol, timeframe, limit=500):
    for exid in ["bybit", "kucoin", "kraken"]:
        try:
            ex = _ex(exid)
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                raw, columns=["time", "open", "high", "low", "close", "volume"]
            )
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df = df.set_index("time")

            print(f"Debug: {symbol} {timeframe} raw data: {len(df)} bars")

            if df["open"].dtype.kind not in "fc":
                print(f"[FATAL] {exid} returned HTML instead of real OHLCV → skipping")
                continue

            df = df.sort_index()
            now = pd.Timestamp.utcnow()
            df = df[df.index <= now + pd.Timedelta(minutes=2)]
            df = df[~((df["volume"] == 0) & (df.index != df.index[-1]))]
            df = df[~df.index.duplicated(keep="last")]

            print(f"Debug: After cleanup: {len(df)} bars")

            if df is None or df.empty or len(df) < 3:
                print(f"[FATAL] Cleaned DF is empty for {symbol} {timeframe}")
                continue

            return df
        except Exception as e:
            print(f"Debug: {exid} failed for {symbol}: {e}")
            continue

    raise RuntimeError(f"No exchange responded for {symbol}")

def _req(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ================== MACRO & DERIVATIVES ==================
def get_yf_quote(t):
    try:
        y = yf.Ticker(t)
        return {
            "last": y.fast_info.last_price,
            "chg_pct": round(
                (y.fast_info.last_price - y.fast_info.previous_close)
                / y.fast_info.previous_close
                * 100,
                2,
            ),
        }
    except Exception:
        return {"last": None, "chg_pct": None}

def get_macro():
    dxy = get_yf_quote("DX-Y.NYB")
    ndx = get_yf_quote("^NDX")
    tnx = get_yf_quote("^TNX")
    if tnx["last"]:
        tnx["last"] = round(tnx["last"] / 10, 2)
    return {"DXY": dxy, "NASDAQ": ndx, "US10Y": tnx}

def get_derivatives(symbols=("BTCUSDT", "ETHUSDT")):
    base = "https://fapi.binance.com"
    out = {}
    for s in symbols:
        oi = _req(f"{base}/fapi/v1/openInterest", {"symbol": s})
        fund = _req(f"{base}/fapi/v1/premiumIndex", {"symbol": s})
        ratio = _req(
            f"{base}/fapi/v1/futures/data/globalLongShortAccountRatio",
            {"symbol": s, "period": "1h", "limit": 1},
        )
        out[s] = {
            "open_interest": float(oi["openInterest"]) if oi else None,
            "funding_rate": float(fund["lastFundingRate"]) if fund else None,
            "long_short_ratio": float(ratio[-1]["longShortRatio"]) if ratio else None,
        }
        time.sleep(0.15)
    return out

def get_onchain_netflow_coinglass(asset="BTC"):
    sym = asset.upper()
    url = "https://open-api.coinglass.com/public/v2/exchange/flow"
    headers = {"accept": "application/json", "coinglassSecret": "free"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        rows = data.get("data", [])
        if not rows:
            return None

        for row in rows:
            if row.get("symbol", "").upper() == sym:
                inflow = float(row.get("inflow", 0))
                outflow = float(row.get("outflow", 0))
                return inflow - outflow
        return None
    except Exception:
        return None

def get_onchain_netflow_binance(asset="BTC"):
    symbol = asset.upper() + "USDT"
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    try:
        d = requests.get(url, timeout=10).json()
        quote_vol = float(d.get("quoteVolume", 0))
        price = float(d.get("lastPrice", 0)) or 1
        base_vol = quote_vol / price
        buy = float(d.get("takerBuyBaseAssetVolume", 0))
        sell = base_vol - buy
        return buy - sell
    except Exception:
        return None

def get_onchain_flow(asset="BTC"):
    cg = get_onchain_netflow_coinglass(asset)
    if cg is not None:
        return cg
    bn = get_onchain_netflow_binance(asset)
    if bn is not None:
        return bn
    return None

def get_orderbook_pressure(symbol="BTCUSDT"):
    url = "https://fapi.binance.com/fapi/v1/depth"
    d = _req(url, {"symbol": symbol, "limit": 50})
    if not d:
        return None
    bids = sum(float(x[1]) for x in d["bids"])
    asks = sum(float(x[1]) for x in d["asks"])
    ratio = round(bids / (asks + 1e-9), 3)
    return {"bids": bids, "asks": asks, "pressure": ratio}

def market_context():
    """
    Market Context PRO
    """
    def _safe_get(url, params=None):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    fg7 = _safe_get("https://api.alternative.me/fng/?limit=7")

    fng_today = None
    fng_avg7 = None

    if fg7 and "data" in fg7:
        vals = []
        for item in fg7["data"]:
            try:
                vals.append(int(item["value"]))
            except Exception:
                pass
        if vals:
            fng_today = vals[0]
            if len(vals) >= 7:
                fng_avg7 = sum(vals) / len(vals)

    f_btc = _safe_get(
        "https://fapi.binance.com/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"}
    )
    f_eth = _safe_get(
        "https://fapi.binance.com/fapi/v1/premiumIndex", {"symbol": "ETHUSDT"}
    )

    fr_btc = None
    fr_eth = None
    if f_btc and "lastFundingRate" in f_btc:
        fr_btc = float(f_btc["lastFundingRate"])
    if f_eth and "lastFundingRate" in f_eth:
        fr_eth = float(f_eth["lastFundingRate"])

    if fr_btc is not None and fr_eth is not None:
        funding_avg = (fr_btc + fr_eth) / 2
    else:
        funding_avg = None

    social = _safe_get(
        "https://lunarcrush.com/api3/correlate/assets",
        {"symbol": "BTC", "data_points": 7},
    )

    social_change = None
    if social and isinstance(social, list):
        try:
            today = social[-1]["social_volume"]
            prev = social[-2]["social_volume"]
            if prev > 0:
                social_change = (today - prev) / prev * 100
        except Exception:
            pass

    macro = {
        "DXY": get_yf_quote("DX-Y.NYB"),
        "NASDAQ": get_yf_quote("^NDX"),
        "US10Y": get_yf_quote("^TNX"),
    }

    derivatives = get_derivatives(("BTCUSDT", "ETHUSDT"))

    onchain = {
        "BTC": get_onchain_flow("BTC"),
        "ETH": get_onchain_flow("ETH"),
    }

    orderbook = {
        "BTC": get_orderbook_pressure("BTCUSDT"),
        "ETH": get_orderbook_pressure("ETHUSDT"),
    }

    return {
        "fng_alt": fng_today,
        "fng_alt_avg7": fng_avg7,
        "funding_rate_avg": funding_avg,
        "social_change": social_change,
        "macro": macro,
        "derivatives": derivatives,
        "onchain": onchain,
        "orderbook": orderbook,
    }

# ================== ANALYSIS WRAPPER ==================
def analyze_timeframe_from_df(
    symbol: str, timeframe: str, df: pd.DataFrame, btc_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    تحلیل کامل یک تایم‌فریم با تمام اندیکاتورها و بخش‌ها
    """
    if df is None or df.empty:
        print(f"‼️ No valid data for {symbol} {timeframe}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_quality": {"status": "bad", "reason": "no_data"},
        }

    if len(df) < 60:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": None,
            "rsi14": None,
            "atr14": None,
            "ema20": None,
            "ema50": None,
            "ema200": None,
            "adx14": None,
            "candle": None,
            "data_quality": {"status": "bad", "reason": "insufficient_data"},
        }

    if len(df) < 2:
        print(f"Warning: Only {len(df)} bar(s) for {symbol} {timeframe}. Skipping.")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": float(df["close"].iloc[-1]) if len(df) >= 1 else None,
            "data_quality": {"status": "warning", "reason": "only_one_bar"},
        }

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    quality = assess_data_quality(df)
    close = df["close"]
    out = {"symbol": symbol, "timeframe": timeframe, "data_quality": quality}

    df = find_pivots(df, left=2, right=2)

    if len(df) >= 20:
        df["ema20"] = ema(close, 20)
    else:
        df["ema20"] = close

    if len(df) >= 50:
        df["ema50"] = ema(close, 50)
    else:
        df["ema50"] = close

    if len(df) >= 200:
        df["ema200"] = ema(close, 200)
    else:
        df["ema200"] = close

    if len(df) >= 14:
        df["rsi14"] = rsi(close, 14)
        df["atr14"] = atr(df, 14)
        df["adx14"] = adx(df, 14)
    else:
        df["rsi14"] = 50
        df["atr14"] = 0
        df["adx14"] = 0

    try:
        vpvr_df = df.tail(min(300, len(df)))
        poc, v_low, v_high = vbp_poc(vpvr_df)
    except Exception:
        poc, v_low, v_high = None, None, None

    bb_ma, bb_up, bb_lo, bb_width = bollinger_bands(close, n=20, k=2.0)
    if bb_ma is not None:
        df["bb_mid"] = bb_ma
        df["bb_upper"] = bb_up
        df["bb_lower"] = bb_lo
        df["bb_width"] = bb_width
    else:
        df["bb_mid"] = close
        df["bb_upper"] = close
        df["bb_lower"] = close
        df["bb_width"] = 0

    df["vol_z"] = volume_zscore(df["volume"], window=20)

    swings, structure_tag, choch, bos = label_structure_from_pivots(df)

    structure_zones = build_structure_zones_from_swings(swings)
    trend_slope_val, trend_slope_label = compute_trendline_slope(close, window=50)
    liquidity_map = liquidity_map_from_vpvr(df, bins=24, top_n=3)
    candle_patterns = detect_candle_patterns(df, lookback=5)

    volume_climax = False
    if not pd.isna(df["vol_z"].iloc[-1]):
        try:
            volume_climax = float(df["vol_z"].iloc[-1]) >= 2.5
        except Exception:
            volume_climax = False

    if len(df) < 3:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_quality": {"status": "bad", "reason": "insufficient_rows"},
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    confirmed = (last["close"] > last["ema20"]) and (last["volume"] > prev["volume"])
    price = float(last["close"])
    atr_rel = float(last["atr14"] / price) if price > 0 and last["atr14"] is not None else None
    bbw = float(last["bb_width"]) if not pd.isna(last["bb_width"]) else None
    vol_z = float(last["vol_z"]) if not pd.isna(last["vol_z"]) else None

    ema_slope = None
    if len(df) >= 5:
        try:
            ema_slope = (
                (df["ema20"].iloc[-1] - df["ema20"].iloc[-5])
                / df["ema20"].iloc[-5]
            ) * 100
        except (IndexError, ZeroDivisionError):
            ema_slope = None

    regime = classify_volatility(
        bbw,
        atr_rel,
        float(last["adx14"]) if not pd.isna(last["adx14"]) else None,
        ema_slope,
    )

    last_high = next((s["price"] for s in reversed(swings) if s["type"] == "H"), None)
    last_low = next((s["price"] for s in reversed(swings) if s["type"] == "L"), None)

    pivot_distances = {}
    if last_high and price:
        pivot_distances["from_last_high_pct"] = float(
            (price - last_high) / last_high * 100
        )
    if last_low and price:
        pivot_distances["from_last_low_pct"] = float(
            (price - last_low) / last_low * 100
        )

    fvg_list = detect_fvg(df)

    low_volume_flag = False
    if len(df) >= 50:
        median_volume = df["volume"].tail(50).median()
        if median_volume > 0:
            low_volume_flag = float(last["volume"]) < median_volume * 0.1

    price_for_dev = df["close"].iloc[-1]

    if len(df) >= 90:
        high_90d = df["high"].rolling(90).max().iloc[-1]
        low_90d = df["low"].rolling(90).min().iloc[-1]
    else:
        high_90d = df["high"].max()
        low_90d = df["low"].min()

    out["dist_from_90d_high_pct"] = round(
        (price_for_dev - high_90d) / high_90d * 100, 4
    )
    out["dist_from_90d_low_pct"] = round(
        (price_for_dev - low_90d) / low_90d * 100, 4
    )

    # === INTEGRATE NEW MODULES ===
    if MODULES_AVAILABLE:
        try:
            # Velocity
            velocity_signal = compute_velocity_signal(df)
            out["velocity"] = velocity_signal
            
            # Trend Quality
            trend_quality = trend_quality_score(df)
            out["trend_quality"] = trend_quality
            
            # Liquidity Engine
            liquidity_heatmap = compute_liquidity_heatmap(df)
            out["liquidity_heatmap"] = liquidity_heatmap
            
            # Market Phase
            market_phase = detect_market_phase(df)
            out["market_phase"] = market_phase
            
            # Risk Structure
            risk_structure = risk_structure_score(df)
            out["risk_structure"] = risk_structure
            
            # SR Reliability
            sr_reliability = compute_sr_reliability(df)
            out["sr_reliability"] = sr_reliability
            
            # Compression
            compression_signal = compute_compression_signal(df)
            out["compression"] = compression_signal
            
            # Dollar Volume
            dollar_volume = compute_dollar_volume(df)
            out["dollar_volume"] = dollar_volume
            
        except Exception as e:
            print(f"⚠️ Error in new modules for {symbol} {timeframe}: {e}")
    else:
        print(f"⚠️ New modules not available for {symbol} {timeframe}")

    # Breadth (only for daily timeframe)
    if timeframe in ("1d", "1D", "D") and MODULES_AVAILABLE:
        try:
            breadth_summary = compute_breadth_summary(df)
            out["breadth"] = breadth_summary
        except Exception as e:
            print(f"⚠️ Error in breadth module for {symbol} {timeframe}: {e}")

    out.update(
        {
            "price": price,
            "rsi14": float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None,
            "atr14": float(last["atr14"]) if not pd.isna(last["atr14"]) else None,
            "ema20": float(last["ema20"]) if not pd.isna(last["ema20"]) else None,
            "ema50": float(last["ema50"]) if not pd.isna(last["ema50"]) else None,
            "ema200": float(last["ema200"]) if not pd.isna(last["ema200"]) else None,
            "adx14": float(last["adx14"]) if not pd.isna(last["adx14"]) else None,
            "vpvr_poc": float(poc) if poc is not None else None,
            "vpvr_low": float(v_low) if v_low is not None else None,
            "vpvr_high": float(v_high) if v_high is not None else None,
            "fvg": fvg_list,
            "candle": {
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": price,
                "volume": float(last["volume"]),
                "ema20": float(last["ema20"]) if not pd.isna(last["ema20"]) else None,
                "ema50": float(last["ema50"]) if not pd.isna(last["ema50"]) else None,
                "ema200": float(last["ema200"]) if not pd.isna(last["ema200"]) else None,
                "confirmed": bool(confirmed),
                "low_volume_flag": bool(low_volume_flag),
            },
            "volatility": {
                "bb_mid": float(last["bb_mid"]) if not pd.isna(last["bb_mid"]) else None,
                "bb_upper": float(last["bb_upper"])
                if not pd.isna(last["bb_upper"])
                else None,
                "bb_lower": float(last["bb_lower"])
                if not pd.isna(last["bb_lower"])
                else None,
                "bb_width": bbw,
                "atr_rel": atr_rel,
                "volume_zscore": vol_z,
                "ema_slope": float(ema_slope) if ema_slope is not None else None,
                "regime": regime,
                "volume_climax": bool(volume_climax),
            },
            "structure": {
                "last_swings": [
                    {
                        "type": s["type"],
                        "price": float(s["price"]),
                        "time": s["time"].isoformat(),
                    }
                    for s in swings
                ],
                "tag": structure_tag,
                "choch": choch,
                "bos": bos,
                "pivot_distances": pivot_distances,
            },
            "structure_zones": structure_zones,
            "trendline": {"slope_pct": trend_slope_val, "direction": trend_slope_label},
            "liquidity_map": liquidity_map,
            "candle_patterns_htf": candle_patterns,
            "volume_climax": bool(volume_climax),
        }
    )

    crash_score, crash_flag, crash_reasons = compute_crash_risk(
        price=price,
        adx=float(last["adx14"]) if not pd.isna(last["adx14"]) else None,
        rsi=float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None,
        atr_rel=atr_rel,
        bb_width=bbw,
        ema_slope=ema_slope,
        volume_climax=volume_climax,
        structure_tag=structure_tag,
        choch=choch,
        liquidity_map=liquidity_map,
        fvg_list=fvg_list,
    )

    out["crash_risk"] = {
        "score": crash_score,
        "flag": crash_flag,
        "reasons": crash_reasons,
    }

    if btc_df is None or btc_df.empty:
        print(
            f"[CORR DEBUG] symbol={symbol}, timeframe={timeframe} → btc_df is None or empty → skip corr_beta"
        )
    else:
        print(
            f"[CORR DEBUG] ENTER corr_beta | symbol={symbol}, timeframe={timeframe}, len(df)={len(df)}, len(btc_df)={len(btc_df)}"
        )
        corr, beta = corr_beta(df, btc_df, window=90)
        print(
            f"[CORR DEBUG] EXIT corr_beta | symbol={symbol}, timeframe={timeframe}, corr={corr}, beta={beta}"
        )
        out["correlation"] = {"with_btc": corr, "beta_vs_btc": beta}

    if timeframe in ("1d", "1D", "D"):
        dev90_hi, dev90_lo = dev_from_extremes(df, days=90)
        dev365_hi, dev365_lo = dev_from_extremes(df, days=365)

        out["liquidity"] = {
            "score_30d_notional": liquidity_score(df, days=30),
        }
        out["deviation"] = {
            "from_90d_high_pct": dev90_hi,
            "from_90d_low_pct": dev90_lo,
            "from_365d_high_pct": dev365_hi,
            "from_365d_low_pct": dev365_lo,
        }
    else:
        out["liquidity"] = {"score_30d_notional": None}

    return out

def analyze_timeframe(symbol, timeframe="4h", limit=500):
    df = fetch_ohlcv(symbol, timeframe, limit)
    return analyze_timeframe_from_df(symbol, timeframe, df)

# ================== SAVE REPORT ==================
def save_report(data, ctx, out_dir):
    """
    data: {
        "symbol": "BTCUSUT",
        "4h": { ... },
        "1d": { ... }
    }
    """
    import pandas as pd

    # Generate / refresh market CSVs with offline fallback
    heatmap_csv_path = "market_heatmap.csv"
    archive_csv_path = "market_archive.csv"
    archive_json_path = "market_archive.json"
    try:
        generate_market_csv(heatmap_csv_path)
    except Exception as e:
        print(f"⚠️ Heatmap generation failed, using cached file if available: {e}")

    try:
        archive_csv_path, archive_json_path = generate_market_archive(
            heatmap_csv=heatmap_csv_path,
            archive_csv=archive_csv_path,
            archive_json=archive_json_path,
        )
    except Exception as e:
        print(f"⚠️ Market archive generation failed, using cached files if available: {e}")

    # -------- PREP --------
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M")

    sym = data.get("symbol", "UNKNOWN")
    safe_sym = sym.replace("/", "_")
    d4 = data.get("4h", {}) or {}
    d1 = data.get("1d", {}) or {}

    # -------- CSV LOADER FUNCTIONS --------
    def load_market_csv_summary(csv_path: str) -> str:
        """
        Convert heatmap CSV to ~10–12 summary lines
        """
        if not csv_path or not os.path.exists(csv_path):
            return "⚠ Market CSV not found."

        try:
            df = pd.read_csv(csv_path)
            cols = ["symbol", "price", "change_24h", "volume_24h"]
            df = df[[c for c in cols if c in df.columns]]

            lines = ["=== MARKET HEATMAP (CSV) ==="]
            for _, row in df.head(12).iterrows():
                s = str(row.get("symbol", ""))
                p = row.get("price", "")
                c = row.get("change_24h", "")
                v = row.get("volume_24h", "")
                lines.append(f"{s:6} | Price: {p} | Change: {c}% | Vol: {v}")

            return "\n".join(lines)
        except Exception as e:
            return f"⚠ Error reading heatmap CSV: {e}"

    def load_archive_csv_summary(csv_path: str) -> str:
        """
        Convert market_archive.csv into summary text
        """
        if not csv_path or not os.path.exists(csv_path):
            return "⚠ Market archive CSV not found."

        try:
            df = pd.read_csv(csv_path)
            lines = ["=== MARKET ARCHIVE (CSV) ==="]
            for _, row in df.iterrows():
                section = str(row.get("section", ""))
                name = str(row.get("name", ""))
                val = row.get("value", "")
                extra = row.get("extra", "")
                if extra and not pd.isna(extra):
                    lines.append(f"{section:12} | {name:12} | {val} | {extra}")
                else:
                    lines.append(f"{section:12} | {name:12} | {val}")
            return "\n".join(lines)
        except Exception as e:
            return f"⚠ Error reading archive CSV: {e}"

    # -------- HELPERS --------
    def g(d, k, default="-"):
        if d is None:
            return default
        v = d.get(k, None)
        if v is None:
            return default
        try:
            return float(v) if isinstance(v, (int, float)) else v
        except Exception:
            return v

    def gcandle(d):
        c = d.get("candle", {}) or {}
        return f"{c}" if c else "-"

    # Context blocks
    m = (ctx or {}).get("macro", {}) or {}
    deriv = (ctx or {}).get("derivatives", {}) or {}
    oc = (ctx or {}).get("onchain", {}) or {}
    ob = (ctx or {}).get("orderbook", {}) or {}
    fg = ctx or {}

    # -------- REPORT BUILDING --------
    md = []
    md.append(f"# {sym} Report ({ts} UTC)\n")

    # 4h TIMEFRAME
    md.append("## 4h timeframe")
    md.append(f"- price: {g(d4, 'price')}")
    md.append(f"- rsi14: {g(d4, 'rsi14')}")
    md.append(f"- atr14: {g(d4, 'atr14')}")
    md.append(f"- ema20: {g(d4, 'ema20')}")
    md.append(f"- ema50: {g(d4, 'ema50')}")
    md.append(f"- ema200: {g(d4, 'ema200')}")
    md.append(f"- adx14: {g(d4, 'adx14')}")

    if "vpvr_poc" in d4 or "vpvr_low" in d4 or "vpvr_high" in d4:
        md.append(
            f"- vpvr_poc: {g(d4, 'vpvr_poc')} (range {g(d4, 'vpvr_low')}–{g(d4, 'vpvr_high')})"
        )

    fvg4 = d4.get("fvg", []) or []
    if fvg4:
        md.append(f"- FVG: {len(fvg4)} gaps detected")

    vol4 = d4.get("volatility", {}) or {}
    md.append(f"- bb_width: {g(vol4, 'bb_width')}")
    md.append(f"- atr_rel: {g(vol4, 'atr_rel')}")
    md.append(f"- ema_slope: {g(vol4, 'ema_slope')}")
    md.append(f"- volatility_regime: {vol4.get('regime', '-')}")

    struct4 = d4.get("structure", {}) or {}
    md.append(f"- structure: {struct4.get('tag', '-')}")
    md.append(f"- CHOCH: {struct4.get('choch', False)}")
    md.append(f"- BOS: {struct4.get('bos', False)}")

    zones4 = d4.get("structure_zones", []) or []
    trend4 = d4.get("trendline", {}) or {}
    liq_map4 = d4.get("liquidity_map", []) or []
    patterns4 = d4.get("candle_patterns_htf", []) or []

    md.append(f"- structure_zones: {len(zones4)}")
    if zones4:
        zone_lvls = [round(z["level"], 2) for z in zones4[:4]]
        md.append(f"- key_s_r_levels: {zone_lvls}")

    md.append(f"- trendline_slope_pct: {g(trend4, 'slope_pct')}")
    md.append(f"- trendline_direction: {trend4.get('direction', '-')}")

    md.append(f"- liquidity_nodes: {len(liq_map4)}")
    if liq_map4:
        liq_levels = [
            f"{round(z['price_low'], 2)}–{round(z['price_high'], 2)} ({round(z['volume_share_pct'], 1)}%)"
            for z in liq_map4[:3]
        ]
        md.append(f"- liquidity_map_top: {liq_levels}")

    vol_climax4 = vol4.get("volume_climax", d4.get("volume_climax", False))
    md.append(f"- volume_climax: {bool(vol_climax4)}")

    if patterns4:
        md.append(f"- candle_patterns_htf: {[p.get('pattern') for p in patterns4]}")
    else:
        md.append("- candle_patterns_htf: []")

    # NEW MODULES FOR 4H
    if MODULES_AVAILABLE:
        md.append("\n=== NEW MODULES (4h) ===")
        md.append(f"Compression: {g(d4, 'compression')}")
        md.append(f"SR Reliability: {g(d4, 'sr_reliability')}")
        md.append(f"Market Phase: {g(d4, 'market_phase')}")
        md.append(f"Risk Structure: {g(d4, 'risk_structure')}")
        md.append(f"Trend Quality: {g(d4, 'trend_quality')}")
        md.append(f"Velocity: {g(d4, 'velocity')}")
        md.append(f"Liquidity: {g(d4, 'liquidity_heatmap')}")
        md.append(f"Dollar Volume: {g(d4, 'dollar_volume')}")

    cr4 = d4.get("crash_risk", {}) or {}
    md.append(f"- crash_risk_score: {cr4.get('score', 'N/A')}")
    md.append(f"- crash_risk_flag: {cr4.get('flag', False)}")
    md.append(f"- crash_reasons: {cr4.get('reasons', [])}")

    corr4 = d4.get("correlation", {}) or {}
    md.append(f"- corr_btc: {g(corr4, 'with_btc')}")
    md.append(f"- beta_btc: {g(corr4, 'beta_vs_btc')}")

    md.append(f"- candle: {gcandle(d4)}")
    md.append(f"- data_quality: {d4.get('data_quality', {}).get('status', '-')}")
    md.append(f"- dist_from_90d_high: {g(d4, 'dist_from_90d_high_pct')}%")
    md.append(f"- dist_from_90d_low: {g(d4, 'dist_from_90d_low_pct')}%\n")

    # 1d TIMEFRAME
    md.append("## 1d timeframe")
    md.append(f"- price: {g(d1, 'price')}")
    md.append(f"- rsi14: {g(d1, 'rsi14')}")
    md.append(f"- atr14: {g(d1, 'atr14')}")
    md.append(f"- ema20: {g(d1, 'ema20')}")
    md.append(f"- ema50: {g(d1, 'ema50')}")
    md.append(f"- ema200: {g(d1, 'ema200')}")
    md.append(f"- adx14: {g(d1, 'adx14')}")
    md.append(f"- dist_from_90d_high: {g(d1, 'dist_from_90d_high_pct')}%")
    md.append(f"- dist_from_90d_low: {g(d1, 'dist_from_90d_low_pct')}%")

    if "vpvr_poc" in d1:
        md.append(
            f"- vpvr_poc: {g(d1, 'vpvr_poc')} (range {g(d1, 'vpvr_low')}–{g(d1, 'vpvr_high')})"
        )

    fvg1 = d1.get("fvg", [])
    if fvg1:
        md.append(f"- FVG: {len(fvg1)} gaps detected")

    vol1 = d1.get("volatility", {}) or {}
    md.append(f"- bb_width: {g(vol1, 'bb_width')}")
    md.append(f"- atr_rel: {g(vol1, 'atr_rel')}")
    md.append(f"- ema_slope: {g(vol1, 'ema_slope')}")
    md.append(f"- volatility_regime: {vol1.get('regime', '-')}")

    struct1 = d1.get("structure", {}) or {}
    md.append(f"- structure: {struct1.get('tag', '-')}")
    md.append(f"- CHOCH: {struct1.get('choch', False)}")
    md.append(f"- BOS: {struct1.get('bos', False)}")

    zones1 = d1.get("structure_zones", []) or []
    trend1 = d1.get("trendline", {}) or {}
    liq_map1 = d1.get("liquidity_map", []) or []
    patterns1 = d1.get("candle_patterns_htf", []) or []

    md.append(f"- structure_zones: {len(zones1)}")
    if zones1:
        zone_lvls1 = [round(z["level"], 2) for z in zones1[:4]]
        md.append(f"- key_s_r_levels: {zone_lvls1}")

    md.append(f"- trendline_slope_pct: {g(trend1, 'slope_pct')}")
    md.append(f"- trendline_direction: {trend1.get('direction', '-')}")

    md.append(f"- liquidity_nodes: {len(liq_map1)}")
    if liq_map1:
        liq_levels1 = [
            f"{round(z['price_low'], 2)}–{round(z['price_high'], 2)} ({round(z['volume_share_pct'], 1)}%)"
            for z in liq_map1[:3]
        ]
        md.append(f"- liquidity_map_top: {liq_levels1}")

    vol_climax1 = vol1.get("volume_climax", d1.get("volume_climax", False))
    md.append(f"- volume_climax: {bool(vol_climax1)}")

    if patterns1:
        md.append(f"- candle_patterns_htf: {[p.get('pattern') for p in patterns1]}")
    else:
        md.append("- candle_patterns_htf: []")

    # NEW MODULES FOR 1D
    if MODULES_AVAILABLE:
        md.append("\n=== NEW MODULES (1d) ===")
        md.append(f"Compression: {g(d1, 'compression')}")
        md.append(f"SR Reliability: {g(d1, 'sr_reliability')}")
        md.append(f"Market Phase: {g(d1, 'market_phase')}")
        md.append(f"Risk Structure: {g(d1, 'risk_structure')}")
        md.append(f"Trend Quality: {g(d1, 'trend_quality')}")
        md.append(f"Velocity: {g(d1, 'velocity')}")
        md.append(f"Liquidity: {g(d1, 'liquidity_heatmap')}")
        md.append(f"Dollar Volume: {g(d1, 'dollar_volume')}")
        md.append(f"Market Breadth: {g(d1, 'breadth')}")

    cr1 = d1.get("crash_risk", {}) or {}
    md.append(f"- crash_risk_score: {cr1.get('score', 'N/A')}")
    md.append(f"- crash_risk_flag: {cr1.get('flag', False)}")
    md.append(f"- crash_reasons: {cr1.get('reasons', [])}")

    corr1 = d1.get("correlation", {}) or {}
    md.append(f"- corr_btc: {g(corr1, 'with_btc')}")
    md.append(f"- beta_btc: {g(corr1, 'beta_vs_btc')}")

    liq1 = d1.get("liquidity", {}) or {}
    dev1 = d1.get("deviation", {}) or {}
    md.append(f"- liquidity_30d: {g(liq1, 'score_30d_notional')}")
    md.append(f"- dist_from_365d_high: {g(dev1, 'from_365d_high_pct')}%")
    md.append(f"- dist_from_365d_low: {g(dev1, 'from_365d_low_pct')}%")

    md.append(f"- candle: {gcandle(d1)}")
    md.append(f"- data_quality: {d1.get('data_quality', {}).get('status', '-')}\n")

    # MACRO
    md.append("---\n## 🌍 Macro Context")
    md.append(
        f"- DXY: {m.get('DXY', {}).get('last', 'N/A')} ({m.get('DXY', {}).get('chg_pct', 'N/A')}%)"
    )
    md.append(
        f"- NASDAQ100: {m.get('NASDAQ', {}).get('last', 'N/A')} ({m.get('NASDAQ', {}).get('chg_pct', 'N/A')}%)"
    )
    md.append(f"- US10Y: {m.get('US10Y', {}).get('last', 'N/A')}%")

    fng_today = fg.get("fng_alt")
    fng_avg7 = fg.get("fng_alt_avg7")

    if fng_today is not None:
        fg_txt = f"{fng_today} (7d avg: {round(fng_avg7,2) if fng_avg7 else 'N/A'})"
    else:
        fg_txt = "N/A"

    md.append(f"- Fear & Greed: {fg_txt}")

    # DERIVATIVES
    md.append("## 📊 Derivatives")
    md.append(f"- BTC OI: {(deriv.get('BTCUSDT') or {}).get('open_interest')}")
    md.append(f"- BTC Funding: {(deriv.get('BTCUSDT') or {}).get('funding_rate')}")
    md.append(
        f"- BTC L/S Ratio: {(deriv.get('BTCUSDT') or {}).get('long_short_ratio')}"
    )
    md.append(f"- ETH OI: {(deriv.get('ETHUSDT') or {}).get('open_interest')}")
    md.append(f"- ETH Funding: {(deriv.get('ETHUSDT') or {}).get('funding_rate')}")
    md.append(
        f"- ETH L/S Ratio: {(deriv.get('ETHUSDT') or {}).get('long_short_ratio')}\n"
    )

    # ON-CHAIN
    md.append("## 🔗 On-chain Flow")
    btc_net = oc.get("BTC")
    eth_net = oc.get("ETH")

    md.append(f"- BTC Netflow: {btc_net if btc_net is not None else 'N/A'}")
    md.append(f"- ETH Netflow: {eth_net if eth_net is not None else 'N/A'}\n")

    # ORDERBOOK
    md.append("## 🧭 Orderbook Pressure")
    md.append(f"- BTC: {ob.get('BTC')}")
    md.append(f"- ETH: {ob.get('ETH')}\n")

    # MARKET CSV + ARCHIVE
    md.append("\n---\n## 📈 Market Heatmap (CSV)")
    md.append(load_market_csv_summary(heatmap_csv_path))

    md.append("\n---\n## 🏦 Market Archive (CSV)")
    md.append(load_archive_csv_summary(archive_csv_path))

    md_text = "\n".join(md)

    md_path = os.path.join(out_dir, f"{safe_sym}_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    json_path = os.path.join(out_dir, f"{safe_sym}_{ts}.json")
    payload = {"symbol": sym, "timeframes": {"4h": d4, "1d": d1}, "context": ctx or {}}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Report saved: {md_path}")
    return md_path

# ================== CLI ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    ap.add_argument("--report_dir", default="reports")
    args = ap.parse_args()

    ctx = market_context()

    btc_symbol = "BTC/USDT"
    print("📡 Fetching BTC reference data...")
    try:
        btc_4h = fetch_ohlcv(btc_symbol, "4h", limit=1000)
        btc_1d = fetch_ohlcv(btc_symbol, "1d", limit=600)
    except Exception as e:
        print(f"❌ Failed to fetch BTC data: {e}")
        btc_4h = None
        btc_1d = None

    for s in [x.strip() for x in args.symbols.split(",") if x.strip()]:
        print(f"📡 Scanning {s} ...")
        try:
            data_4h_df = fetch_ohlcv(s, "4h", 500)
            data_1d_df = fetch_ohlcv(s, "1d", 500)

            data_4h = analyze_timeframe_from_df(s, "4h", data_4h_df, btc_4h)
            data_1d = analyze_timeframe_from_df(s, "1d", data_1d_df, btc_1d)

            merged = {"symbol": s, "4h": data_4h, "1d": data_1d}
            save_report(merged, ctx, args.report_dir)
        except Exception as e:
            print(f"❌ Failed to analyze {s}: {e}")
            continue
