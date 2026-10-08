"""
oos_metrics.py

Pure functions for the out-of-sample (OOS) performance test.
There is no Streamlit code in this file, so every function can also be
called and tested from Spyder.

Overview of the logic (details are in the comments below):
  1. clean_weights      -> tidy weight table per (date, strategy, asset)
  2. prepare_prices     -> wide price table (rows = dates, columns = assets)
  3. build_asset_map    -> link asset names in the weight file to price columns
  4. run_strategy_backtest / run_oos_test
                        -> daily OOS returns after each recommendation date
  5. performance_metrics, benchmark_metrics, drawdown_series, monthly_table
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters (all defaults can be changed from the app sidebar)
# ---------------------------------------------------------------------------
@dataclass
class OosParams:
    # Trading days between the last price date on or before the recommendation
    # date and the purchase at the close. 1 = buy at the close of the next day.
    exec_lag: int = 1
    # "buy_and_hold": weights drift between recommendation dates.
    # "constant_mix": weights are reset to the target every period.
    holding_mode: str = "buy_and_hold"
    # What happens to weight on assets without a usable price:
    # "cash" (kept as cash with 0 % return) or "renormalize" (spread over the rest).
    unmatched_mode: str = "cash"
    # Transaction cost in basis points of traded notional.
    cost_bps: float = 0.0
    # Recommendation dates before this date are ignored (None = use all).
    oos_start: pd.Timestamp | None = None
    # Last return day that is evaluated (None = last price date).
    oos_end: pd.Timestamp | None = None
    # A recommendation is skipped if the purchase date lies more than this many
    # calendar days after the recommendation date.
    max_base_gap_days: int = 7


# Metric labels with their display type. The app uses these for formatting.
METRIC_KINDS = {
    "First return day": "date",
    "Last return day": "date",
    "Observations": "int",
    "Cumulative return": "pct",
    "Annualized return": "pct",
    "Annualized volatility": "pct",
    "Sharpe ratio": "ratio",
    "Sortino ratio": "ratio",
    "Omega ratio": "ratio",
    "Maximum drawdown": "pct",
    "Calmar ratio": "ratio",
    "VaR 95% per period (historical)": "pct",
    "CVaR 95% per period (historical)": "pct",
    "Share of positive periods": "pct",
    "Best period": "pct",
    "Worst period": "pct",
}
BENCH_KINDS = {
    "Benchmark cumulative return": "pct",
    "Excess cumulative return": "pct",
    "Tracking error (annualized)": "pct",
    "Information ratio": "ratio",
    "Beta": "ratio",
    "Jensen's alpha (annualized)": "pct",
    "Treynor ratio (annualized)": "pct",
}


# ---------------------------------------------------------------------------
# 1. Weights
# ---------------------------------------------------------------------------
def clean_weights(raw, date_col, strategy_col, asset_col, weight_col,
                  est_end_col=None, source_name="", zero_tol=1e-12,
                  dayfirst=False):
    """
    Return (tidy, diagnostics, n_bad_rows).

    tidy has the columns date, strategy, asset, weight, est_end.
    Weights are rescaled so that they sum to 1 for every (date, strategy).
    diagnostics lists the weight sum before rescaling and the number of
    positions per (date, strategy).
    """
    # Step 1: build a table with standard column names
    strategy = (raw[strategy_col].astype(str).str.strip() if strategy_col
                else pd.Series(source_name, index=raw.index))
    est_end = (pd.to_datetime(raw[est_end_col], errors="coerce", dayfirst=dayfirst).dt.normalize()
               if est_end_col else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]"))
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=dayfirst).dt.normalize(),
        "strategy": strategy,
        "asset": raw[asset_col].astype(str).str.strip(),
        "weight": pd.to_numeric(raw[weight_col], errors="coerce"),
        "est_end": est_end,
    })

    # Step 2: drop rows that cannot be used and count them
    bad = df[["date", "asset", "weight"]].isna().any(axis=1) | (df["asset"] == "")
    n_bad = int(bad.sum())
    df = df.loc[~bad]

    # Step 3: collapse duplicate (date, strategy, asset) rows by summing weights
    keys = ["date", "strategy", "asset"]
    df = df.groupby(keys, as_index=False, sort=True).agg(
        weight=("weight", "sum"), est_end=("est_end", "max"))

    # Step 4: remove numerical residuals (for example -1e-17) and zero weights
    df.loc[df["weight"].abs() <= zero_tol, "weight"] = 0.0
    df = df.loc[df["weight"] != 0.0].copy()

    # Step 5: rescale to a sum of 1 per (date, strategy) and keep the old sum
    grp = df.groupby(["date", "strategy"])["weight"]
    sums = grp.transform("sum")
    diag = (df.assign(sum_before=sums)
              .groupby(["date", "strategy"], as_index=False)
              .agg(sum_before=("sum_before", "first"), positions=("asset", "size")))
    df["weight"] = np.where(sums != 0, df["weight"] / sums, np.nan)
    df = df.dropna(subset=["weight"]).reset_index(drop=True)
    return df, diag, n_bad


# ---------------------------------------------------------------------------
# 2. Prices
# ---------------------------------------------------------------------------
def prepare_prices(raw, layout, date_col, asset_col=None, price_col=None,
                   ffill_limit=5, dayfirst=False):
    """
    Return (wide, info). wide has one row per date and one column per asset.
    layout is "wide" (one column per asset) or "long" (date, asset, price).
    """
    # Step 1: convert the upload to a wide table
    if layout == "wide":
        dates = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=dayfirst)
        wide = raw.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
        wide.index = pd.DatetimeIndex(dates).normalize()
    else:
        long = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=dayfirst).dt.normalize(),
            "asset": raw[asset_col].astype(str).str.strip(),
            "price": pd.to_numeric(raw[price_col], errors="coerce"),
        }).dropna()
        wide = long.pivot_table(index="date", columns="asset", values="price", aggfunc="last")

    # Step 2: tidy the index and the columns
    wide = wide.loc[~wide.index.isna()]
    wide = wide.loc[~wide.index.duplicated(keep="last")].sort_index()
    wide.columns = [str(c).strip() for c in wide.columns]
    wide = wide.loc[:, ~pd.Index(wide.columns).duplicated()]

    # Step 3: prices must be positive numbers
    n_invalid = int((wide <= 0).sum().sum())
    wide = wide.where(wide > 0)
    wide = wide.dropna(axis=1, how="all")

    # Step 4: carry the last price forward over short gaps (holidays, missing cells)
    if ffill_limit and ffill_limit > 0:
        wide = wide.ffill(limit=int(ffill_limit))

    info = {
        "dates": len(wide),
        "assets": wide.shape[1],
        "first": wide.index.min() if len(wide) else pd.NaT,
        "last": wide.index.max() if len(wide) else pd.NaT,
        "invalid_cells": n_invalid,
    }
    return wide, info


# ---------------------------------------------------------------------------
# 3. Matching asset names
# ---------------------------------------------------------------------------
def _strip_suffix(name):
    """AAPL.O -> AAPL (text after the last dot is treated as exchange suffix)."""
    return str(name).strip().upper().rsplit(".", 1)[0]


def build_asset_map(weight_assets, price_columns, ignore_suffix=False, explicit=None):
    """
    Map every asset name in the weight files to a price column (or None).
    Order of rules: explicit mapping table, exact match ignoring case,
    optional match ignoring the exchange suffix (only if the result is unique).
    """
    explicit = explicit or {}
    exact = {str(c).strip().upper(): c for c in price_columns}
    by_stem = {}
    if ignore_suffix:
        groups = {}
        for c in price_columns:
            groups.setdefault(_strip_suffix(c), []).append(c)
        by_stem = {k: v[0] for k, v in groups.items() if len(v) == 1}

    mapping = {}
    for a in weight_assets:
        key = str(a).strip().upper()
        if key in explicit and explicit[key] in price_columns:
            mapping[a] = explicit[key]
        elif key in exact:
            mapping[a] = exact[key]
        elif ignore_suffix and _strip_suffix(a) in by_stem:
            mapping[a] = by_stem[_strip_suffix(a)]
        else:
            mapping[a] = None
    return mapping


# ---------------------------------------------------------------------------
# 4. Out-of-sample backtest
# ---------------------------------------------------------------------------
PERIOD_COLUMNS = [
    "Recommendation date", "Purchase date", "First return day", "Last return day",
    "Holdings used", "Weight matched", "Unmatched holdings", "Carried-forward prices",
    "One-way turnover", "Cost", "Period return", "Estimation window check", "Status",
]


def _skipped_row(rec_date, reason):
    row = {c: np.nan for c in PERIOD_COLUMNS}
    row["Recommendation date"] = rec_date
    row["Status"] = "skipped: " + reason
    return row


def run_strategy_backtest(w, prices, params):
    """
    Evaluate the weights of ONE strategy against the price table.

    w needs the columns date, col (price column or None), weight, est_end.
    Returns (daily_returns, periods).

    Timing: the last price date on or before the recommendation date is the
    signal day s. The portfolio is bought at the close of day s + exec_lag.
    Returns are earned from the next price date on, until the close of the
    purchase day of the next recommendation (or the last evaluated day).
    """
    idx = prices.index
    n = len(idx)
    last_idx = n - 1
    if params.oos_end is not None:
        last_idx = min(last_idx, int(idx.searchsorted(params.oos_end, side="right")) - 1)

    # Step 1: find the purchase day of every recommendation date
    plan, skipped = [], []
    rec_dates = sorted(w["date"].unique())
    if params.oos_start is not None:
        rec_dates = [d for d in rec_dates if d >= params.oos_start]
    for d in rec_dates:
        signal = int(idx.searchsorted(d, side="right")) - 1
        buy = signal + int(params.exec_lag)
        if buy < 0 or buy + 1 > last_idx:
            skipped.append(_skipped_row(d, "no price data after the purchase date"))
        elif (idx[buy] - d).days > params.max_base_gap_days:
            skipped.append(_skipped_row(d, "purchase date too far after recommendation date"))
        else:
            plan.append((d, buy))

    # Step 2: if two recommendations map to the same purchase day, keep the later one
    dedup = {}
    for d, buy in plan:
        if buy in dedup:
            skipped.append(_skipped_row(dedup[buy], "replaced by a later recommendation on the same day"))
        dedup[buy] = d
    plan = [(d, b) for b, d in sorted(dedup.items())]

    pieces, rows = [], []
    prev_end = {}            # drifted weights at the end of the previous period

    # Step 3: evaluate every holding period
    for k, (d, buy) in enumerate(plan):
        seg_end = plan[k + 1][1] if k + 1 < len(plan) else last_idx
        seg_end = min(seg_end, last_idx)

        # 3a: weights of this recommendation, merged onto price columns
        rows_d = w.loc[w["date"] == d]
        mapped = (rows_d.dropna(subset=["col"]).groupby("col", as_index=False)["weight"].sum())
        base_px = prices.iloc[buy].reindex(mapped["col"]).to_numpy()
        usable = mapped.loc[np.isfinite(base_px)]
        total = float(rows_d["weight"].sum())
        usable_sum = float(usable["weight"].sum())
        coverage = usable_sum / total if total else np.nan
        n_unmatched = int(len(rows_d) - len(usable))

        # 3b: decide how unmatched weight is treated
        wi = usable["weight"].to_numpy(dtype=float)
        if params.unmatched_mode == "renormalize" and usable_sum != 0:
            wi = wi * (total / usable_sum)
            cash = 0.0
        else:
            cash = total - usable_sum
        cols = list(usable["col"])
        if not cols:
            skipped.append(_skipped_row(d, "no holding has a price on the purchase date"))
            continue

        # 3c: relative price path since the purchase day
        px = prices.iloc[buy:seg_end + 1][cols]
        carried = int(px.isna().sum().sum())
        growth = (px / px.iloc[0]).ffill().to_numpy(dtype=float)

        # 3d: daily portfolio returns
        if params.holding_mode == "constant_mix":
            asset_ret = growth[1:] / growth[:-1] - 1.0
            r = asset_ret @ wi
            end_w = dict(zip(cols, wi))
        else:
            value = cash + growth @ wi
            bad = np.where(value <= 0)[0]
            if bad.size:                      # only possible with short positions
                value = value[:bad[0]]        # cut the period before the value reaches zero
            r = value[1:] / value[:-1] - 1.0
            end_w = dict(zip(cols, wi * growth[len(value) - 1] / value[-1]))
        if len(r) == 0:
            skipped.append(_skipped_row(d, "no valid return days"))
            continue

        # 3e: turnover and cost at the purchase
        union = set(end_w) | set(prev_end) | set(cols)
        new_w = dict(zip(cols, wi))
        delta = sum(abs(new_w.get(c, 0.0) - prev_end.get(c, 0.0)) for c in union)
        cost = params.cost_bps / 1e4 * delta
        r[0] = (1.0 + r[0]) * (1.0 - cost) - 1.0
        prev_end = end_w

        # 3f: store the daily returns and the period summary
        dates_r = idx[buy + 1: buy + 1 + len(r)]
        pieces.append(pd.Series(r, index=dates_r))
        est_end = rows_d["est_end"].max()
        if pd.isna(est_end):
            check = "n/a"
        else:
            check = "OK" if dates_r[0] > est_end else "FAIL: returns inside estimation window"
        rows.append({
            "Recommendation date": d,
            "Purchase date": idx[buy],
            "First return day": dates_r[0],
            "Last return day": dates_r[-1],
            "Holdings used": len(cols),
            "Weight matched": coverage,
            "Unmatched holdings": n_unmatched,
            "Carried-forward prices": carried,
            "One-way turnover": delta / 2.0,
            "Cost": cost,
            "Period return": float(np.prod(1.0 + r) - 1.0),
            "Estimation window check": check,
            "Status": "ok" if len(dates_r) == seg_end - buy else "ok (cut)",
        })

    returns = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    periods = pd.DataFrame(rows + skipped, columns=PERIOD_COLUMNS)
    if len(periods):
        periods = periods.sort_values("Recommendation date").reset_index(drop=True)
    return returns, periods


def run_oos_test(weights, prices, asset_map, params):
    """
    Run the test for every strategy in the tidy weight table.
    Returns (returns_by_strategy, periods).
    """
    weights = weights.copy()
    weights["col"] = weights["asset"].map(asset_map)
    series, periods = {}, []
    for strat, grp in weights.groupby("strategy", sort=False):
        ret, per = run_strategy_backtest(grp, prices, params)
        per.insert(0, "Strategy", strat)
        periods.append(per)
        if len(ret):
            series[strat] = ret
    returns = pd.DataFrame(series).sort_index() if series else pd.DataFrame()
    periods = pd.concat(periods, ignore_index=True) if periods else pd.DataFrame()
    return returns, periods


def benchmark_returns(prices, column):
    """Simple period returns of one price column."""
    return prices[column].pct_change()


# ---------------------------------------------------------------------------
# 5. Performance statistics
# ---------------------------------------------------------------------------
def infer_periods_per_year(index):
    """Guess the return frequency from the median gap between dates."""
    if len(index) < 3:
        return 252
    gap = pd.Series(index).diff().dt.days.median()
    if gap <= 4:
        return 252
    if gap <= 8:
        return 52
    if gap <= 35:
        return 12
    if gap <= 100:
        return 4
    return 1


def _div(a, b):
    return a / b if (b is not None and np.isfinite(b) and b != 0) else np.nan


def drawdown_series(ret):
    """Drawdown from the running peak of the wealth index (start value 1)."""
    wealth = (1.0 + ret).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return wealth / peak - 1.0


def performance_metrics(ret, ppy, rf_annual=0.0):
    """Return a dict with the entries of METRIC_KINDS."""
    r = ret.dropna().astype(float)
    out = {k: np.nan for k in METRIC_KINDS}
    n = len(r)
    if n == 0:
        return out
    out["First return day"], out["Last return day"], out["Observations"] = r.index[0], r.index[-1], n
    cum = float(np.prod(1.0 + r.to_numpy()) - 1.0)
    out["Cumulative return"] = cum
    out["Best period"], out["Worst period"] = float(r.max()), float(r.min())
    out["Share of positive periods"] = float((r > 0).mean())
    if n < 2:
        return out

    rf_p = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0        # risk-free rate per period
    ex = r - rf_p
    out["Annualized return"] = (1.0 + cum) ** (ppy / n) - 1.0 if 1.0 + cum > 0 else np.nan
    out["Annualized volatility"] = float(r.std(ddof=1) * np.sqrt(ppy))
    out["Sharpe ratio"] = _div(ex.mean(), ex.std(ddof=1)) * np.sqrt(ppy)
    downside = float(np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2)))
    out["Sortino ratio"] = _div(ex.mean(), downside) * np.sqrt(ppy)
    out["Omega ratio"] = _div(np.maximum(ex, 0).sum(), np.maximum(-ex, 0).sum())
    mdd = float(drawdown_series(r).min())
    out["Maximum drawdown"] = mdd
    out["Calmar ratio"] = _div(out["Annualized return"], abs(mdd))
    var = float(np.percentile(r, 5))
    out["VaR 95% per period (historical)"] = var
    out["CVaR 95% per period (historical)"] = float(r[r <= var].mean())
    return out


def benchmark_metrics(ret, bench, ppy, rf_annual=0.0):
    """Benchmark-relative statistics on the dates both series share."""
    out = {k: np.nan for k in BENCH_KINDS}
    pair = pd.concat([ret.rename("r"), bench.rename("b")], axis=1).dropna()
    if len(pair) < 2:
        return out
    r, b = pair["r"], pair["b"]
    rf_p = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0
    cum_r = float(np.prod(1.0 + r.to_numpy()) - 1.0)
    cum_b = float(np.prod(1.0 + b.to_numpy()) - 1.0)
    diff = r - b
    te = float(diff.std(ddof=1) * np.sqrt(ppy))
    beta = _div(r.cov(b), b.var(ddof=1))
    excess_r, excess_b = r - rf_p, b - rf_p
    out["Benchmark cumulative return"] = cum_b
    out["Excess cumulative return"] = cum_r - cum_b
    out["Tracking error (annualized)"] = te
    out["Information ratio"] = _div(diff.mean() * ppy, te)
    out["Beta"] = beta
    out["Jensen's alpha (annualized)"] = (excess_r.mean() - beta * excess_b.mean()) * ppy
    out["Treynor ratio (annualized)"] = _div(excess_r.mean() * ppy, beta)
    return out


def monthly_table(ret):
    """Compounded returns by calendar month, plus a total per year."""
    r = ret.dropna()
    if r.empty:
        return pd.DataFrame()
    months = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    table = months.unstack()
    names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    table = table.rename(columns=names)
    table["Year (compounded)"] = (1.0 + r).groupby(r.index.year).prod() - 1.0
    table.index.name = "Year"
    return table


# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-08)  New file. Added: OosParams, clean_weights, prepare_prices,
#   build_asset_map, run_strategy_backtest, run_oos_test, benchmark_returns,
#   infer_periods_per_year, drawdown_series, performance_metrics,
#   benchmark_metrics, monthly_table, METRIC_KINDS, BENCH_KINDS.
