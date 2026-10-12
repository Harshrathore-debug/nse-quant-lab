"""Performance and statistical-significance metrics.

Everything takes a series of *simple* daily returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

TD = 252


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    years = len(r) / TD
    return float((1 + r).prod() ** (1 / years) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(TD))


def sharpe(r: pd.Series, rf: pd.Series | None = None) -> float:
    x = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(TD))


def sortino(r: pd.Series, rf: pd.Series | None = None) -> float:
    x = (r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)).dropna()
    downside = np.sqrt((np.minimum(x, 0) ** 2).mean())
    return float(x.mean() / downside * np.sqrt(TD))


def drawdown(r: pd.Series) -> pd.Series:
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1


def max_dd(r: pd.Series) -> float:
    return float(drawdown(r).min())


def calmar(r: pd.Series) -> float:
    return cagr(r) / abs(max_dd(r))


def alpha_beta(r: pd.Series, bench: pd.Series, rf: pd.Series, lags: int = 5) -> dict:
    """CAPM regression on excess returns with Newey-West (HAC) standard errors."""
    df = pd.concat([r - rf, bench - rf], axis=1, keys=["y", "x"]).dropna()
    model = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_ann": float(model.params["const"] * TD),
        "alpha_t": float(model.tvalues["const"]),
        "beta": float(model.params["x"]),
        "corr": float(df["y"].corr(df["x"])),
    }


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P(true SR > benchmark), non-normal adjusted.

    Sharpe ratios here are per-period (daily), benchmark given annualised.
    """
    r = r.dropna()
    n = len(r)
    sr = r.mean() / r.std()
    sr_b = sr_benchmark / np.sqrt(TD)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, trial_sharpes: np.ndarray) -> float:
    """DSR (Bailey & Lopez de Prado 2014): PSR against the max Sharpe expected
    from N independent trials with zero true skill. trial_sharpes are annualised."""
    n_trials = len(trial_sharpes)
    var_sr = np.var(np.asarray(trial_sharpes) / np.sqrt(TD), ddof=1)
    emc = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - emc) * z1 + emc * z2)
    return probabilistic_sharpe(r, sr0 * np.sqrt(TD))


def bootstrap_sharpe_ci(r: pd.Series, n_boot: int = 2000, block: int = 20, seed: int = 7,
                        ci: float = 0.95) -> tuple[float, float]:
    """Stationary-ish block bootstrap CI for the annualised Sharpe ratio."""
    x = r.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        out[i] = sample.mean() / sample.std() * np.sqrt(TD)
    lo, hi = np.quantile(out, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def summary(r: pd.Series, rf: pd.Series, bench: pd.Series | None = None, name: str | None = None) -> dict:
    out = {
        "name": name or r.name,
        "cagr": cagr(r),
        "vol": ann_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "max_dd": max_dd(r),
        "calmar": calmar(r),
        "total_return": float((1 + r.fillna(0)).prod() - 1),
        "start": str(r.dropna().index[0].date()),
        "end": str(r.dropna().index[-1].date()),
    }
    if bench is not None:
        out.update(alpha_beta(r, bench, rf))
    return out


def table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("name")
    pct = ["cagr", "vol", "max_dd", "total_return", "alpha_ann"]
    fmt = df.copy()
    for c in df.columns:
        if c in pct:
            fmt[c] = (df[c] * 100).map("{:.1f}%".format)
        elif df[c].dtype.kind == "f":
            fmt[c] = df[c].map("{:.2f}".format)
    return fmt
