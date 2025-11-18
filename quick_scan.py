import os
import sys
import json
import math
import time
import textwrap
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
import pandas as pd


BASE_URL = "https://api.bybit.com"
HEATMAP_CSV = "market_heatmap.csv"
MARKET_ARCHIVE_CSV = "market_archive.csv"

SCAN_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "TRX/USDT",
    "ADA/USDT",
]

FEAR_GREED_URL = "https://api.alternative.me/fng"
COINGLASS_DERIV_URL = "https://open-api.coinglass.com/api/pro/v1/futures/openInterest"
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY", "")

CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", "")
CRYPTOQUANT_BASE = "https://api.cryptoquant.com/v1/btc"


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}")


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fetch_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    try:
        resp = requests.get(url, params=params or {}, headers=headers or {}, timeout=10)
        if resp.status_code != 200:
            log(f"⚠️ HTTP {resp.status_code} for {url}")
            return None
        return resp.json()
    except Exception as e:
        log(f"❌ fetch_json error for {url}: {e}")
        return None


def fetch_ohlcv(symbol: str, interval: str = "240", limit: int = 200) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Bybit (spot) for a symbol like 'BTC/USDT'.
    """
    try:
        base, quote = symbol.split("/")
        url = f"{BASE_URL}/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": f"{base}{quote}",
            "interval": interval,
            "limit": limit,
        }
        js = fetch_json(url, params=params)
        if not js or js.get("retCode") != 0:
            log(f"⚠️ Bybit kline error for {symbol}: {js}")
            return None

        rows = js.get("result", {}).get("list", [])
        if not rows:
            return None

        # Bybit returns newest first
        rows = list(reversed(rows))
        df = pd.DataFrame(
            rows,
            columns=[
                "startTime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ],
        )
        df["startTime"] = pd.to_datetime(df["startTime"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        log(f"❌ fetch_ohlcv error for {symbol} {interval}: {e}")
        return None


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_market_heatmap_row(symbol: str, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> Dict[str, Any]:
    row = {
        "symbol": symbol,
        "price_4h": None,
        "change_4h_pct": None,
        "price_1d": None,
        "change_1d_pct": None,
        "rsi_4h": None,
        "rsi_1d": None,
        "trend_4h": None,
        "trend_1d": None,
    }
    try:
        if df_4h is not None and not df_4h.empty:
            close4 = df_4h["close"].iloc[-1]
            row["price_4h"] = close4
            if len(df_4h) > 1:
                row["change_4h_pct"] = (close4 / df_4h["close"].iloc[-2] - 1) * 100
            rsi4 = rsi(df_4h["close"], 14).iloc[-1]
            row["rsi_4h"] = rsi4
            ema_fast4 = ema(df_4h["close"], 20).iloc[-1]
            ema_slow4 = ema(df_4h["close"], 50).iloc[-1]
            row["trend_4h"] = "bullish" if ema_fast4 > ema_slow4 else "bearish"

        if df_1d is not None and not df_1d.empty:
            close1d = df_1d["close"].iloc[-1]
            row["price_1d"] = close1d
            if len(df_1d) > 1:
                row["change_1d_pct"] = (close1d / df_1d["close"].iloc[-2] - 1) * 100
            rsi1d = rsi(df_1d["close"], 14).iloc[-1]
            row["rsi_1d"] = rsi1d
            ema_fast1d = ema(df_1d["close"], 20).iloc[-1]
            ema_slow1d = ema(df_1d["close"], 50).iloc[-1]
            row["trend_1d"] = "bullish" if ema_fast1d > ema_slow1d else "bearish"

    except Exception as e:
        log(f"⚠️ build_market_heatmap_row error for {symbol}: {e}")
    return row


def generate_market_csv(symbols: List[str], out_dir: str = ".") -> str:
    rows = []
    for s in symbols:
        log(f"📡 Fetching 4h data for heatmap: {s}")
        df4 = fetch_ohlcv(s, interval="240", limit=100)
        log(f"📡 Fetching 1d data for heatmap: {s}")
        df1 = fetch_ohlcv(s, interval="D", limit=100)
        row = build_market_heatmap_row(s, df4, df1)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, HEATMAP_CSV)
    df.to_csv(csv_path, index=False)
    log(f"✅ Market heatmap saved to {csv_path}")
    return csv_path


def load_market_csv_summary(csv_path: str) -> str:
    if not os.path.exists(csv_path):
        return ""
    df = pd.read_csv(csv_path)
    cols = ["symbol", "price_4h", "change_4h_pct", "price_1d", "change_1d_pct", "rsi_4h", "rsi_1d", "trend_4h", "trend_1d"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    out = df[cols].to_csv(index=False)
    return out.strip()


def update_market_archive(symbols: List[str], out_dir: str = ".") -> str:
    """
    Append latest heatmap snapshot to MARKET_ARCHIVE_CSV with timestamp.
    """
    csv_path = os.path.join(out_dir, HEATMAP_CSV)
    if not os.path.exists(csv_path):
        return ""

    df = pd.read_csv(csv_path)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df["timestamp_utc"] = ts

    archive_path = os.path.join(out_dir, MARKET_ARCHIVE_CSV)
    if os.path.exists(archive_path):
        old = pd.read_csv(archive_path)
        df_all = pd.concat([old, df], ignore_index=True)
    else:
        df_all = df.copy()

    df_all.to_csv(archive_path, index=False)
    log(f"📚 Market archive updated at {archive_path}")
    return archive_path


def fetch_fear_greed() -> Dict[str, Any]:
    js = fetch_json(FEAR_GREED_URL, params={"limit": 1})
    if not js or "data" not in js or not js["data"]:
        return {}
    d = js["data"][0]
    return {
        "value": d.get("value"),
        "value_classification": d.get("value_classification"),
        "timestamp": d.get("timestamp"),
    }


def fetch_derivatives_snapshot() -> Dict[str, Any]:
    if not COINGLASS_API_KEY:
        return {}
    headers = {"coinglassSecret": COINGLASS_API_KEY}
    params = {"symbol": "BTC"}
    js = fetch_json(COINGLASS_DERIV_URL, params=params, headers=headers)
    if not js or js.get("code") != 0:
        return {}
    data = js.get("data", [])
    if not data:
        return {}
    total_oi = sum([safe_get(x, "openInterest", default=0) for x in data])
    return {
        "open_interest_usd": total_oi,
    }


def fetch_cryptoquant_metric(metric: str) -> Any:
    if not CRYPTOQUANT_API_KEY:
        return None
    url = f"{CRYPTOQUANT_BASE}/indicator/{metric}"
    params = {"api_key": CRYPTOQUANT_API_KEY}
    js = fetch_json(url, params=params)
    return js


def fetch_macro_context() -> Dict[str, Any]:
    """
    Aggregate macro context: Fear & Greed, BTC DOM, DXY, rates, basic derivatives & on-chain.
    This is intentionally lightweight but robust.
    """
    ctx = {
        "macro": {},
        "derivatives_1d": {},
        "derivatives_7d": {},
        "onchain": {},
        "orderbook": {},
    }

    # Fear & Greed
    fng = fetch_fear_greed()
    if fng:
        ctx["macro"]["fear_greed"] = {
            "value": fng.get("value"),
            "classification": fng.get("value_classification"),
            "timestamp": fng.get("timestamp"),
        }

    # Placeholder BTC DOM / DXY / rates
    ctx["macro"]["btc_dominance"] = {
        "value": None,
        "change_24h": None,
    }
    ctx["macro"]["dxy"] = {
        "value": None,
        "change_24h": None,
    }
    ctx["macro"]["rates"] = {
        "fed_funds": {
            "value": None,
            "bias": None,
        }
    }

    # Derivatives basic snapshot
    deriv = fetch_derivatives_snapshot()
    if deriv:
        ctx["derivatives_1d"] = {
            "open_interest_usd": deriv.get("open_interest_usd"),
            "oi_change_pct": None,
            "funding_rate": None,
        }
        ctx["derivatives_7d"] = {
            "open_interest_usd": deriv.get("open_interest_usd"),
            "oi_change_pct": None,
        }

    # Light on-chain and orderbook placeholders
    ctx["onchain"] = {
        "active_addresses_24h": None,
        "tx_count_24h": None,
    }
    ctx["orderbook"] = {
        "btc_imbalance": None,
        "spot_volume_24h": None,
    }

    return ctx


def analyze_timeframe_from_df(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """
    Analyze a timeframe DataFrame: trend, volatility, pullbacks, basic structure.
    """
    if df is None or df.empty:
        return {"label": label, "valid": False}

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]

    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    atr14 = atr(df, 14)
    rsi14 = rsi(closes, 14)

    last = len(df) - 1
    if last < 1:
        return {"label": label, "valid": False}

    trend = "bullish" if ema20.iloc[last] > ema50.iloc[last] else "bearish"
    price = closes.iloc[last]
    atr_val = atr14.iloc[last]
    rsi_val = rsi14.iloc[last]

    body = abs(df["close"].iloc[last] - df["open"].iloc[last])
    range_candle = df["high"].iloc[last] - df["low"].iloc[last]
    body_ratio = body / (range_candle + 1e-9)

    vol = df["volume"].rolling(20).mean().iloc[last]
    vol_prev = df["volume"].rolling(20).mean().iloc[last - 1] if last >= 1 else vol

    pullback = None
    recent_high = highs.rolling(20).max().iloc[last]
    recent_low = lows.rolling(20).min().iloc[last]
    if trend == "bullish":
        pullback = (recent_high - price) / (recent_high + 1e-9) * 100
    else:
        pullback = (price - recent_low) / (recent_low + 1e-9) * 100

    return {
        "label": label,
        "valid": True,
        "trend": trend,
        "price": price,
        "atr": atr_val,
        "rsi": rsi_val,
        "body_ratio": body_ratio,
        "avg_vol": vol,
        "avg_vol_prev": vol_prev,
        "pullback_pct": pullback,
        "recent_high": recent_high,
        "recent_low": recent_low,
    }


def format_timeframe_summary(tf: Dict[str, Any]) -> List[str]:
    if not tf.get("valid"):
        return [f"- {tf.get('label')} data not available"]

    trend = tf["trend"]
    price = tf["price"]
    rsi_val = tf["rsi"]
    pullback = tf["pullback_pct"]
    atr_val = tf["atr"]

    lines = []
    lines.append(f"- Trend: **{trend.upper()}** at ~**{price:.2f}**")
    lines.append(f"- RSI(14): **{rsi_val:.1f}**")
    lines.append(f"- ATR(14): **{atr_val:.2f}**")
    lines.append(f"- Pullback vs recent extreme: **{pullback:.2f}%**")
    return lines


def save_report(merged: Dict[str, Any], ctx: Dict[str, Any], out_dir: str = ".") -> str:
    """
    Save a detailed markdown report for a single symbol.
    Also appends a compact heatmap + archive summary at the bottom.
    """
    os.makedirs(out_dir, exist_ok=True)

    symbol = merged.get("symbol", "UNKNOWN")
    base, quote = symbol.split("/")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    md_name = f"{base}{quote}_{ts}_report.md"
    md_path = os.path.join(out_dir, md_name)

    tf4 = merged.get("tf_4h", {})
    tf1 = merged.get("tf_1d", {})

    macro = ctx.get("macro", {})
    d1 = ctx.get("derivatives_1d", {})
    d7 = ctx.get("derivatives_7d", {})
    onchain = ctx.get("onchain", {})
    orderbook = ctx.get("orderbook", {})

    lines = []
    lines.append(f"# {symbol} Report ({ts} UTC)")
    lines.append("")
    lines.append("## 4h timeframe")
    lines.append("")
    for ln in format_timeframe_summary(tf4):
        lines.append(ln)
    lines.append("")
    lines.append("## 1d timeframe")
    lines.append("")
    for ln in format_timeframe_summary(tf1):
        lines.append(ln)

    # Basic volatility & structure
    lines.append("")
    lines.append("## Volatility & Structure")
    lines.append("")
    if tf4.get("valid") and tf1.get("valid"):
        lines.append(
            f"- 4h ATR(14): **{tf4['atr']:.2f}**, 1d ATR(14): **{tf1['atr']:.2f}**"
        )
        lines.append(
            f"- 4h pullback: **{tf4['pullback_pct']:.2f}%**, 1d pullback: **{tf1['pullback_pct']:.2f}%**"
        )
    else:
        lines.append("- Not enough data for full volatility comparison.")

    # Macro snapshot (per symbol view)
    lines.append("")
    lines.append("## Macro + Derivatives + On-chain & Orderbook Snapshot")
    lines.append("")

    fng = macro.get("fear_greed", {})
    if fng:
        lines.append(
            f"- **Crypto Fear & Greed**: {fng.get('value', 'N/A')} / 100 ({fng.get('classification', 'N/A')})"
        )
    btc_dom = macro.get("btc_dominance", {})
    if btc_dom:
        lines.append(
            f"- **BTC Dominance**: {btc_dom.get('value', 'N/A')}% (Δ {btc_dom.get('change_24h', 'N/A')}%)"
        )
    dxy = macro.get("dxy", {})
    if dxy:
        lines.append(
            f"- **DXY**: {dxy.get('value', 'N/A')} (Δ {dxy.get('change_24h', 'N/A')}%)"
        )
    rates = macro.get("rates", {})
    if rates:
        fed = rates.get("fed_funds", {})
        if fed:
            lines.append(
                f"- **Fed Funds Rate**: {fed.get('value', 'N/A')}% ({fed.get('bias', 'N/A')})"
            )

    lines.append("")
    lines.append("### Derivatives")
    if d1:
        lines.append(
            f"- **Open Interest 24h**: {d1.get('open_interest_usd', 'N/A')} USD (Δ {d1.get('oi_change_pct', 'N/A')}%)"
        )
        lines.append(
            f"- **Funding Rate 24h**: {d1.get('funding_rate', 'N/A')}%"
        )
    if d7:
        lines.append(
            f"- **Open Interest 7d**: {d7.get('open_interest_usd', 'N/A')} USD (Δ {d7.get('oi_change_pct', 'N/A')}%)"
        )

    lines.append("")
    lines.append("### On-chain & Liquidity")
    if onchain:
        lines.append(
            f"- **Active Addresses (24h)**: {onchain.get('active_addresses_24h', 'N/A')}"
        )
        lines.append(
            f"- **TX Count (24h)**: {onchain.get('tx_count_24h', 'N/A')}"
        )
    if orderbook:
        lines.append(
            f"- **Orderbook Imbalance (BTC)**: {orderbook.get('btc_imbalance', 'N/A')}"
        )
        lines.append(
            f"- **Spot CEX Volume 24h**: {orderbook.get('spot_volume_24h', 'N/A')}"
        )

    # Save markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Attach compact heatmap + archive summary to the same symbol report
    try:
        heatmap_summary = load_market_csv_summary(os.path.join(out_dir, HEATMAP_CSV))
        archive_path = os.path.join(out_dir, MARKET_ARCHIVE_CSV)
        archive_exists = os.path.exists(archive_path)

        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n\n---\n")
            f.write("## 📈 Market Heatmap (CSV)\n\n")
            if heatmap_summary:
                f.write("```csv\n")
                f.write(heatmap_summary)
                f.write("\n```\n")
            else:
                f.write("_No heatmap CSV available_\n")

            f.write("\n## 🏦 Market Archive (CSV)\n\n")
            if archive_exists:
                with open(archive_path, "r", encoding="utf-8") as af:
                    f.write("```csv\n")
                    f.write(af.read().strip())
                    f.write("\n```\n")
            else:
                f.write("_No archive CSV available_\n")

    except Exception as e:
        log(f"⚠️ Could not append heatmap/archive summary to {md_path}: {e}")

    log(f"✅ Report saved to {md_path}")
    return md_path


ALL_REPORT_PATH = None  # global path for combined multi-symbol report


def append_to_all_report(symbol_md_path: str, ctx: dict, report_dir: str = ".") -> None:
    """
    Append a single-symbol markdown report into a global ALL_REPORTS_*.txt file.

    - On first call, creates the ALL_REPORT file with:
      * Header + timestamp
      * Market heatmap CSV
      * Market archive CSV (if present)
      * Global macro / derivatives / on-chain / orderbook snapshot
      * "Per-Symbol Reports" section header

    - On every call, appends only the per-symbol part of the report
      (without duplicating the heatmap/footer section).
    """
    import os
    from datetime import datetime

    global ALL_REPORT_PATH

    # Determine or create ALL_REPORT file on first call
    if ALL_REPORT_PATH is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
        ALL_REPORT_PATH = os.path.join(report_dir, f"ALL_REPORTS_{ts}.txt")

        # Load heatmap + archive CSVs
        heatmap_path = os.path.join(report_dir, HEATMAP_CSV)
        archive_path = os.path.join(report_dir, MARKET_ARCHIVE_CSV)

        heatmap_csv = ""
        if os.path.exists(heatmap_path):
            try:
                # Re-use existing helper for compact summary
                heatmap_csv = load_market_csv_summary(heatmap_path)
            except Exception:
                with open(heatmap_path, "r", encoding="utf-8") as hf:
                    heatmap_csv = hf.read().strip()

        archive_csv = ""
        if os.path.exists(archive_path):
            try:
                with open(archive_path, "r", encoding="utf-8") as af:
                    archive_csv = af.read().strip()
            except Exception:
                archive_csv = ""

        # Unpack context safely
        macro = ctx.get("macro", {}) or {}
        d1 = ctx.get("derivatives_1d", {}) or {}
        d7 = ctx.get("derivatives_7d", {}) or {}
        onchain = ctx.get("onchain", {}) or {}
        orderbook = ctx.get("orderbook", {}) or {}

        fng = macro.get("fear_greed", {}) or {}
        btc_dom = macro.get("btc_dominance", {}) or {}
        dxy = macro.get("dxy", {}) or {}
        rates = macro.get("rates", {}) or {}

        lines = []
        lines.append(f"# 🧠 ALL REPORT — {ts} UTC")
        lines.append("")
        lines.append("## 🌐 Market Heatmap (CSV)")
        lines.append("")
        if heatmap_csv:
            lines.append("```csv")
            lines.append(heatmap_csv)
            lines.append("```")
        else:
            lines.append("_No heatmap CSV available_")
        lines.append("")
        lines.append("## 🏦 Market Archive (CSV)")
        lines.append("")
        if archive_csv:
            lines.append("```csv")
            lines.append(archive_csv)
            lines.append("```")
        else:
            lines.append("_No archive CSV available_")
        lines.append("")
        lines.append("## 🌍 Macro Context")
        lines.append("")
        if macro:
            if fng:
                lines.append(
                    f"- **Crypto Fear & Greed**: {fng.get('value', 'N/A')} / 100 ({fng.get('classification', 'N/A')})"
                )
            if btc_dom:
                lines.append(
                    f"- **BTC Dominance**: {btc_dom.get('value', 'N/A')}% (Δ {btc_dom.get('change_24h', 'N/A')}%)"
                )
            if dxy:
                lines.append(
                    f"- **DXY**: {dxy.get('value', 'N/A')} (Δ {dxy.get('change_24h', 'N/A')}%)"
                )
            if rates:
                fed = rates.get("fed_funds", {})
                if fed:
                    lines.append(
                        f"- **Fed Funds Rate**: {fed.get('value', 'N/A')}% ({fed.get('bias', 'N/A')})"
                    )
        else:
            lines.append("- Macro data not available.")

        lines.append("")
        lines.append("## 📉 Derivatives Overview")
        lines.append("")
        if d1 or d7:
            if d1:
                lines.append(
                    f"- **Open Interest 24h**: {d1.get('open_interest_usd', 'N/A')} USD (Δ {d1.get('oi_change_pct', 'N/A')}%)"
                )
                lines.append(
                    f"- **Funding Rate 24h**: {d1.get('funding_rate', 'N/A')}%"
                )
            if d7:
                lines.append(
                    f"- **Open Interest 7d**: {d7.get('open_interest_usd', 'N/A')} USD (Δ {d7.get('oi_change_pct', 'N/A')}%)"
                )
        else:
            lines.append("- Derivatives data not available.")

        lines.append("")
        lines.append("## 🔗 On-chain & Liquidity Snapshot")
        lines.append("")
        if onchain:
            lines.append(
                f"- **Active Addresses (24h)**: {onchain.get('active_addresses_24h', 'N/A')}"
            )
            lines.append(
                f"- **TX Count (24h)**: {onchain.get('tx_count_24h', 'N/A')}"
            )
        else:
            lines.append("- On-chain data not available.")
        if orderbook:
            lines.append(
                f"- **Orderbook Imbalance (BTC)**: {orderbook.get('btc_imbalance', 'N/A')}"
            )
            lines.append(
                f"- **Spot CEX Volume 24h**: {orderbook.get('spot_volume_24h', 'N/A')}"
            )
        else:
            lines.append("- Orderbook data not available.")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("# 📊 Per-Symbol Reports (4h + 1d + modules)")
        lines.append("")

        with open(ALL_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # Append current symbol report (without its local heatmap footer)
    if not os.path.exists(symbol_md_path):
        return

    with open(symbol_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Strip local heatmap section to avoid duplication in ALL_REPORT
    split_marker = "\n---\n## 📈 Market Heatmap (CSV)"
    if split_marker in md_text:
        per_symbol_text = md_text.split(split_marker, 1)[0].rstrip()
    else:
        per_symbol_text = md_text.rstrip()

    with open(ALL_REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n\n---\n")
        f.write(per_symbol_text)
        f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid crypto market scanner + reporter.")
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SCAN_UNIVERSE),
        help="Comma-separated symbols like BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory to save reports",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = SCAN_UNIVERSE[:]

    os.makedirs(args.report_dir, exist_ok=True)

    # Step 1: Generate / refresh heatmap + archive
    heatmap_path = os.path.join(args.report_dir, HEATMAP_CSV)
    if not os.path.exists(heatmap_path):
        generate_market_csv(symbols, out_dir=args.report_dir)
    else:
        log(f"ℹ️ Using existing heatmap CSV at {heatmap_path}")

    archive_path = update_market_archive(symbols, out_dir=args.report_dir)

    # Step 2: Fetch macro context once
    log("📡 Fetching macro context (Fear & Greed, derivatives, on-chain stubs)...")
    ctx = fetch_macro_context()

    # Step 3: Per-symbol detailed scan
    for s in symbols:
        log(f"🔍 Scanning symbol: {s}")
        df4 = fetch_ohlcv(s, interval="240", limit=200)
        df1 = fetch_ohlcv(s, interval="D", limit=200)

        tf4 = analyze_timeframe_from_df(df4, "4h")
        tf1 = analyze_timeframe_from_df(df1, "1d")

        merged = {
            "symbol": s,
            "tf_4h": tf4,
            "tf_1d": tf1,
        }

        md_file = save_report(merged, ctx, args.report_dir)
        try:
            append_to_all_report(md_file, ctx, args.report_dir)
        except Exception as e:
            print(f"⚠️ Failed to append {s} to ALL_REPORT: {e}")

    log("✅ Scan complete.")
