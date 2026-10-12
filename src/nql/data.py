"""Data loading for NSE end-of-day files.

Source: NSE bhavcopy/index EOD data mirrored in the open-source eod2_data
repository (github.com/BennyThadikaran/eod2_data). Stock prices are adjusted for
splits and bonuses; neither stocks nor indices include dividends, so every
comparison in this project is price-return vs price-return (like for like).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)

# NIFTY 50 constituents (mid-2026 composition). Using today's members to test
# history introduces survivorship bias; every module flags this where it matters.
NIFTY50 = [
    "adanient", "adaniports", "apollohosp", "asianpaint", "axisbank", "bajaj-auto",
    "bajfinance", "bajajfinsv", "bel", "bhartiartl", "cipla", "coalindia", "drreddy",
    "eichermot", "grasim", "hcltech", "hdfcbank", "hdfclife", "hindalco",
    "hindunilvr", "icicibank", "indigo", "infy", "itc", "jswsteel", "kotakbank",
    "lt", "m&m", "maruti", "nestleind", "ntpc", "ongc", "powergrid", "reliance",
    "sbilife", "sbin", "shriramfin", "sunpharma", "tataconsum", "tatasteel", "tcs",
    "techm", "titan", "trent", "ultracemco", "wipro",
]

BANKS = [
    "hdfcbank", "icicibank", "kotakbank", "axisbank", "sbin", "indusindbk",
    "bankbaroda", "pnb", "federalbnk", "idfcfirstb", "aubank", "bandhanbnk", "canbk",
]

SECTORS = {
    "Bank": "nifty bank", "IT": "nifty it", "FMCG": "nifty fmcg",
    "Pharma": "nifty pharma", "Auto": "nifty auto", "Metal": "nifty metal",
    "Energy": "nifty energy", "Realty": "nifty realty", "Infra": "nifty infrastructure",
    "Media": "nifty media", "PSU Bank": "nifty psu bank", "Fin Serv": "nifty financial services",
}

TRADING_DAYS = 252


# Corporate actions the source file leaves unadjusted: {name: [(ex_date, price_divisor)]}
UNADJUSTED_SPLITS = {"goldbees": [("2019-12-19", 100.0)]}


def load(name: str, field: str = "Close") -> pd.Series:
    df = pd.read_csv(RAW / f"{name}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    s = pd.to_numeric(df[field], errors="coerce")
    s = s[~s.index.duplicated(keep="last")].dropna()
    for ex_date, div in UNADJUSTED_SPLITS.get(name, []):
        if field in ("Open", "High", "Low", "Close"):
            s[s.index < ex_date] = s[s.index < ex_date] / div
    # Some index files have a long hole in their history (e.g. NIFTY50 Equal Weight is blank from
    # Nov-2015 to Apr-2017). Bridging it would book the whole gap as one day's return, so keep only
    # the data after the last gap longer than 30 calendar days.
    gaps = s.index.to_series().diff().dt.days
    big = gaps[gaps > 30]
    if len(big):
        s = s[s.index >= big.index[-1]]
    s.name = name
    return s


def load_frame(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{name}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def panel(names, start=None, field="Close") -> pd.DataFrame:
    df = pd.concat([load(n, field) for n in names], axis=1)
    if start:
        df = df.loc[start:]
    return df


def nifty(start=None) -> pd.Series:
    s = load("nifty 50")
    return s.loc[start:] if start else s


def risk_free_daily(index: pd.DatetimeIndex, fallback_annual: float = 0.065) -> pd.Series:
    """Daily risk-free return from the NIFTY 1D Rate Index (overnight TREPS).

    Before the index starts (Jun-2016) a flat fallback rate is used.
    """
    level = load("nifty 1d rate index")
    level = level.reindex(level.index.union(index)).ffill().reindex(index)
    rate = level.pct_change()
    rate = rate.fillna((1 + fallback_annual) ** (1 / TRADING_DAYS) - 1)
    rate.iloc[0] = 0.0
    return rate


def log_returns(p: pd.DataFrame | pd.Series):
    return np.log(p).diff()


def save_results(module: str, payload: dict):
    """Persist headline numbers so README/dashboard are generated from code, not typed."""
    import json

    path = OUT / "results"
    path.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (pd.Timestamp,)):
            return str(o.date())
        raise TypeError(type(o))

    (path / f"{module}.json").write_text(json.dumps(payload, indent=2, default=conv))
