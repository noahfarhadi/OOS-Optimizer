"""
oos_app.py

Streamlit page for out-of-sample (OOS) performance tests.

Users upload (1) weight recommendations and (2) a price history. The page
computes realized returns after each recommendation date and reports risk and
return statistics, charts and data checks. All calculations are in
oos_metrics.py. Nothing is stored; uploads live in the memory of the session.

Run locally from an Anaconda prompt:
    streamlit run oos_app.py
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import file_io as fio
import oos_metrics as om

# ---------------------------------------------------------------------------
# Configuration: every default below can be changed here or in the sidebar
# ---------------------------------------------------------------------------
CONFIG = {
    "page_title": "Out-of-sample performance test",
    "intro": ("Upload portfolio recommendations and a price history. The page measures what "
              "each recommendation earned after its date."),
    "disclaimer": ("Results are hypothetical and computed from the uploaded files. They exclude "
                   "taxes, financing and borrowing costs unless entered above, and they do not "
                   "predict future results. This page is informational and is not investment advice."),
    "file_types": ["csv", "txt", "tsv"],
    # Column-name candidates for the automatic column mapping (lower case)
    "date_candidates": ["recommendation_date", "as_of_date", "rebalance_date", "date", "asof", "as_of"],
    "strategy_candidates": ["strategy", "portfolio", "model"],
    "asset_candidates": ["ticker", "asset", "symbol", "security", "ric"],
    "weight_candidates": ["weight", "weights", "allocation", "w"],
    "est_end_candidates": ["estimation_end"],
    "price_candidates": ["adj close", "adj_close", "adjusted_close", "close", "price", "last"],
    # Defaults of the sidebar controls
    "zero_tol": 1e-12,
    "ffill_limit": 5,
    "ignore_suffix": False,
    "exec_lag": 1,
    "max_base_gap_days": 7,
    "cost_bps": 0.0,
    "rf_annual_pct": 0.0,
    "periods_per_year_options": [252, 52, 12, 4, 1],
    "holding_modes": {"Buy and hold (weights drift)": "buy_and_hold",
                      "Constant mix (weights reset every day)": "constant_mix"},
    "unmatched_modes": {"Keep as cash (0 % return)": "cash",
                        "Spread over matched assets": "renormalize"},
    "min_coverage_warning": 0.95,
    "default_strategies_in_chart": 6,
}

# Use the current width argument if this Streamlit version has it
FIT = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
       else {"use_container_width": True})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_table(data: bytes, separator, decimal: str) -> pd.DataFrame:
    """Read an uploaded delimited text file (delimiter and decimal mark are detected by default)."""
    return fio.read_table(data, separator, decimal)


@st.cache_data(show_spinner=False)
def cached_prices(data, separator, decimal, layout, date_col, asset_col, price_col, ffill_limit, dayfirst):
    """Read and prepare the price file once per combination of settings."""
    raw = load_table(data, separator, decimal)
    return om.prepare_prices(raw, layout, date_col, asset_col, price_col, ffill_limit, dayfirst)


guess = fio.guess_column


def pick(label, columns, guessed, key, optional=False, help_text=None):
    """Selectbox that starts on the guessed column; returns None for '(none)'."""
    options = (["(none)"] if optional else []) + list(columns)
    index = options.index(guessed) if guessed in options else 0
    choice = st.selectbox(label, options, index=index, key=key, help=help_text)
    return None if choice == "(none)" else choice


def as_text_dates(df):
    """Format datetime columns as text so tables show plain dates."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def style_metrics(table, kinds):
    """Format a metrics table according to the display type of each column."""
    fmt = {}
    out = table.copy()
    for col, kind in kinds.items():
        if col not in out.columns:
            continue
        if kind == "date":
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            fmt[col] = {"pct": "{:.2%}", "ratio": "{:.2f}", "int": "{:,.0f}"}[kind]
    return out.style.format(fmt, na_rep="n/a")


def wealth_line(returns, price_index, base=100.0):
    """Wealth index (start = base) with a start point on the purchase date."""
    r = returns.dropna()
    loc = max(int(price_index.searchsorted(r.index[0])) - 1, 0)
    x = [price_index[loc]] + list(r.index)
    y = np.r_[base, base * (1.0 + r).cumprod().to_numpy()]
    return x, y


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title=CONFIG["page_title"], layout="wide")
st.title(CONFIG["page_title"])
st.caption(CONFIG["intro"])

# Sidebar part 1: settings that are needed to read the files
with st.sidebar:
    st.header("File and data settings")
    sep_label = st.selectbox("Delimiter", list(fio.SEPARATORS), index=0,
                             help="Auto-detect works for each file separately.")
    separator = fio.SEPARATORS[sep_label]
    decimal = st.selectbox("Decimal mark", fio.DECIMALS, index=0)
    dayfirst = st.checkbox("Dates are day-first (DD/MM/YYYY)", value=False)
    zero_tol = st.number_input("Treat weights up to this size as zero", min_value=0.0,
                               value=float(CONFIG["zero_tol"]), format="%.1e", step=1e-12,
                               help="Removes numerical residuals such as -1e-17 from optimizer output.")
    ffill_limit = st.number_input("Carry prices forward over gaps of up to (days)", min_value=0,
                                  value=int(CONFIG["ffill_limit"]), step=1)
    ignore_suffix = st.checkbox("Match tickers ignoring the exchange suffix (AAPL.O = AAPL)",
                                value=CONFIG["ignore_suffix"])

# Step 1 and 2: uploads
st.subheader("1. Recommendations (weights)")
weight_files = st.file_uploader("One or several delimited files with date, strategy, asset and weight",
                                type=CONFIG["file_types"], accept_multiple_files=True, key="weights")
st.subheader("2. Price history")
price_file = st.file_uploader("One file with prices after the recommendation dates",
                              type=CONFIG["file_types"], key="prices")
mapping_file = st.file_uploader("Optional: ticker mapping (column 1 = name in the weight files, "
                                "column 2 = name in the price file)",
                                type=CONFIG["file_types"], key="mapping")

with st.expander("What the files need to contain"):
    st.markdown(
        "- **Weights:** one row per date, strategy and asset, with a weight. Weights are rescaled to "
        "sum to 1 for every date and strategy, so fractions and percentages both work.\n"
        "- **Prices:** closing prices, preferably adjusted for splits and dividends, all in the same "
        "currency. The page does not convert currencies. Either one column per asset (wide) or one "
        "row per date and asset (long).\n"
        "- **Out-of-sample:** only prices after each recommendation date are used. If the weight file "
        "has an estimation end date, the page checks that the evaluated returns start after it.")

if not weight_files or price_file is None:
    st.info("Upload at least one weight file and one price file to start.")
    st.stop()

# Step 3: map the columns of every weight file and clean the weights
frames, diagnostics = [], []
for i, f in enumerate(weight_files):
    raw = load_table(f.getvalue(), separator, decimal)
    cols = list(raw.columns)
    with st.expander(f"Column mapping for {f.name}", expanded=len(weight_files) == 1):
        c1, c2, c3 = st.columns(3)
        with c1:
            date_col = pick("Recommendation date", cols, guess(cols, CONFIG["date_candidates"]), f"date{i}")
            strat_col = pick("Strategy", cols, guess(cols, CONFIG["strategy_candidates"]), f"strat{i}",
                             optional=True, help_text="If none, the file name is used as strategy name.")
        with c2:
            asset_col = pick("Asset", cols, guess(cols, CONFIG["asset_candidates"]), f"asset{i}")
            weight_col = pick("Weight", cols, guess(cols, CONFIG["weight_candidates"]), f"weight{i}")
        with c3:
            est_col = pick("Estimation end date", cols, guess(cols, CONFIG["est_end_candidates"]), f"est{i}",
                           optional=True, help_text="Used only to check that evaluated returns come after it.")
    try:
        tidy, diag, n_bad = om.clean_weights(raw, date_col, strat_col, asset_col, weight_col, est_col,
                                             source_name=f.name.rsplit(".", 1)[0], zero_tol=zero_tol,
                                             dayfirst=dayfirst)
    except Exception as exc:
        st.error(f"{f.name}: the weight file could not be read with this mapping ({exc}).")
        st.stop()
    if n_bad:
        st.warning(f"{f.name}: {n_bad} rows without a valid date, asset or weight were ignored.")
    tidy["source"] = f.name
    diag["source"] = f.name
    frames.append(tidy)
    diagnostics.append(diag)

weights = pd.concat(frames, ignore_index=True)
if weights.empty:
    st.error("No usable weights were found. Check the column mapping.")
    st.stop()

# Strategy names that occur in more than one file get the file name appended
files_per_strategy = weights.groupby("strategy")["source"].nunique()
clash = files_per_strategy[files_per_strategy > 1].index
mask = weights["strategy"].isin(clash)
weights.loc[mask, "strategy"] = weights.loc[mask, "strategy"] + " [" + weights.loc[mask, "source"] + "]"
weight_diag = pd.concat(diagnostics, ignore_index=True)

# Step 4: map the price file and prepare the price table
raw_prices = load_table(price_file.getvalue(), separator, decimal)
pcols = list(raw_prices.columns)
g_asset = guess(pcols, CONFIG["asset_candidates"])
g_price = guess(pcols, CONFIG["price_candidates"])
with st.expander("Column mapping for the price file", expanded=True):
    layout_label = st.radio("Layout", ["Wide: one column per asset", "Long: one row per date and asset"],
                            index=1 if (g_asset and g_price) else 0, horizontal=True)
    layout = "wide" if layout_label.startswith("Wide") else "long"
    p1, p2, p3 = st.columns(3)
    with p1:
        p_date = pick("Date", pcols, guess(pcols, CONFIG["date_candidates"]), "p_date")
    if layout == "long":
        with p2:
            p_asset = pick("Asset", pcols, g_asset, "p_asset")
        with p3:
            p_price = pick("Price", pcols, g_price, "p_price")
    else:
        p_asset = p_price = None
try:
    prices, price_info = cached_prices(price_file.getvalue(), separator, decimal, layout, p_date,
                                       p_asset, p_price, int(ffill_limit), dayfirst)
except Exception as exc:
    st.error(f"The price file could not be read with this mapping ({exc}).")
    st.stop()
if prices.empty or len(prices) < 3:
    st.error("The price table has fewer than three dates. Check the date column and the layout.")
    st.stop()

# Optional explicit mapping of asset names
explicit = {}
if mapping_file is not None:
    mp = load_table(mapping_file.getvalue(), separator, decimal)
    if mp.shape[1] >= 2:
        explicit = {str(a).strip().upper(): str(b).strip() for a, b in zip(mp.iloc[:, 0], mp.iloc[:, 1])}
    else:
        st.warning("The mapping file needs two columns and was ignored.")

# Sidebar part 2: settings of the test itself
with st.sidebar:
    st.header("Test settings")
    first_rec = weights["date"].min().date()
    oos_start = st.date_input("Ignore recommendations dated before", value=first_rec)
    oos_end = st.date_input("Evaluate returns up to", value=prices.index.max().date(),
                            min_value=prices.index.min().date(), max_value=prices.index.max().date())
    exec_lag = st.number_input("Purchase lag (trading days after the recommendation)", min_value=0,
                               max_value=20, value=int(CONFIG["exec_lag"]), step=1,
                               help="0 buys at the close of the recommendation day, 1 at the close of the next trading day.")
    holding_label = st.radio("Holding rule", list(CONFIG["holding_modes"]), index=0)
    unmatched_label = st.radio("Weight on assets without a price", list(CONFIG["unmatched_modes"]), index=0)
    cost_bps = st.number_input("Transaction cost (basis points of traded value)", min_value=0.0,
                               value=float(CONFIG["cost_bps"]), step=1.0)
    rf_pct = st.number_input("Risk-free rate (% per year)", value=float(CONFIG["rf_annual_pct"]), step=0.1)
    max_gap = st.number_input("Maximum days between recommendation and purchase", min_value=0,
                              value=int(CONFIG["max_base_gap_days"]), step=1)
    ppy_inferred = om.infer_periods_per_year(prices.index)
    ppy_choice = st.selectbox("Periods per year (for annualizing)", ["Auto"] + CONFIG["periods_per_year_options"],
                              help=f"Auto uses {ppy_inferred}, estimated from the spacing of the price dates.")
    ppy = ppy_inferred if ppy_choice == "Auto" else int(ppy_choice)
    bench_options = ["(none)"] + list(prices.columns)
    bench_col = st.selectbox("Benchmark (a column of the price file)", bench_options, index=0)

params = om.OosParams(
    exec_lag=int(exec_lag), holding_mode=CONFIG["holding_modes"][holding_label],
    unmatched_mode=CONFIG["unmatched_modes"][unmatched_label], cost_bps=float(cost_bps),
    oos_start=pd.Timestamp(oos_start), oos_end=pd.Timestamp(oos_end), max_base_gap_days=int(max_gap))
rf = float(rf_pct) / 100.0

# Step 5: run the test
asset_map = om.build_asset_map(weights["asset"].unique(), list(prices.columns), ignore_suffix, explicit)
returns, periods = om.run_oos_test(weights, prices, asset_map, params)

tab_sum, tab_chart, tab_detail, tab_checks = st.tabs(["Summary", "Charts", "Strategy detail", "Data checks"])

# Warnings that apply to the whole result
if not periods.empty:
    low = periods["Weight matched"].lt(CONFIG["min_coverage_warning"]).sum()
    if low:
        st.warning(f"{int(low)} holding periods have less than {CONFIG['min_coverage_warning']:.0%} of the "
                   "weight matched to a price. See the Data checks tab.")
    fails = periods["Estimation window check"].astype(str).str.startswith("FAIL").sum()
    if fails:
        st.error(f"{int(fails)} holding periods include returns inside the estimation window. These are not "
                 "out-of-sample. Increase the purchase lag or remove those recommendations.")

if returns.empty:
    with tab_sum:
        st.error("No out-of-sample returns could be computed. The most likely reasons are that the price "
                 "file ends before the recommendation dates or that the asset names do not match.")
    with tab_checks:
        st.dataframe(as_text_dates(periods))
    st.stop()

bench_returns = om.benchmark_returns(prices, bench_col) if bench_col != "(none)" else None

# Statistics for every strategy
stats = {}
for s in returns.columns:
    row = om.performance_metrics(returns[s], ppy, rf)
    if bench_returns is not None:
        row.update(om.benchmark_metrics(returns[s], bench_returns, ppy, rf))
    stats[s] = row
kinds = dict(om.METRIC_KINDS)
if bench_returns is not None:
    kinds.update(om.BENCH_KINDS)
stats_table = pd.DataFrame(stats).T[list(kinds)]

with tab_sum:
    shortest = int(stats_table["Observations"].min())
    if shortest < ppy:
        st.info(f"The shortest sample has {shortest} return periods, fewer than one year ({ppy}). "
                "Annualized figures scale short samples up and are indicative only.")
    st.dataframe(style_metrics(stats_table, kinds))
    d1, d2 = st.columns(2)
    d1.download_button("Download statistics (CSV)", stats_table.to_csv().encode("utf-8"),
                       file_name="oos_statistics.csv", mime="text/csv")
    d2.download_button("Download returns per period (CSV)", returns.to_csv().encode("utf-8"),
                       file_name="oos_returns.csv", mime="text/csv")

with tab_chart:
    chosen = st.multiselect("Strategies", list(returns.columns),
                            default=list(returns.columns)[:CONFIG["default_strategies_in_chart"]])
    if chosen:
        fig = go.Figure()
        for s in chosen:
            x, y = wealth_line(returns[s], prices.index)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=s))
        if bench_returns is not None:
            first = min(returns[s].dropna().index[0] for s in chosen)
            last = max(returns[s].dropna().index[-1] for s in chosen)
            bx, by = wealth_line(bench_returns.loc[first:last], prices.index)
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", name=f"Benchmark: {bench_col}",
                                     line=dict(color="gray", dash="dash")))
        fig.update_layout(title="Wealth index (start = 100)", hovermode="x unified", height=450,
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, **FIT)

        dd_fig = go.Figure()
        for s in chosen:
            dd = om.drawdown_series(returns[s].dropna())
            dd_fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=s))
        dd_fig.update_layout(title="Drawdown", yaxis_tickformat=".0%", hovermode="x unified", height=350,
                             legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(dd_fig, **FIT)

with tab_detail:
    one = st.selectbox("Strategy", list(returns.columns))
    per = periods.loc[periods["Strategy"] == one].drop(columns="Strategy")
    st.markdown("**Holding periods**")
    st.dataframe(as_text_dates(per).style.format({
        "Weight matched": "{:.1%}", "One-way turnover": "{:.2%}", "Cost": "{:.3%}",
        "Period return": "{:.2%}"}, na_rep=""))
    monthly = om.monthly_table(returns[one])
    if not monthly.empty:
        st.markdown("**Monthly returns (compounded)**")
        st.dataframe(monthly.style.format("{:.2%}", na_rep=""))
    st.download_button("Download holding periods (CSV)", per.to_csv(index=False).encode("utf-8"),
                       file_name="oos_periods.csv", mime="text/csv")

with tab_checks:
    st.markdown("**Price table**")
    st.write(f"{price_info['dates']} dates from {price_info['first']:%Y-%m-%d} to {price_info['last']:%Y-%m-%d}, "
             f"{price_info['assets']} assets, {price_info['invalid_cells']} non-positive prices removed.")
    st.markdown("**Weight files**")
    wd = weight_diag.copy()
    wd["date"] = wd["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(wd.rename(columns={"sum_before": "Weight sum before rescaling", "positions": "Positions"}))
    unmatched = weights.assign(col=weights["asset"].map(asset_map))
    unmatched = unmatched.loc[unmatched["col"].isna()]
    st.markdown("**Assets without a matching price column**")
    if unmatched.empty:
        st.write("All assets in the weight files were matched.")
    else:
        table = (unmatched.groupby("asset").agg(strategies=("strategy", "nunique"), largest_weight=("weight", "max"))
                 .sort_values("largest_weight", ascending=False))
        st.dataframe(table.style.format({"largest_weight": "{:.2%}"}))
    st.markdown("**All holding periods**")
    st.dataframe(as_text_dates(periods).style.format({
        "Weight matched": "{:.1%}", "One-way turnover": "{:.2%}", "Cost": "{:.3%}",
        "Period return": "{:.2%}"}, na_rep=""))

st.divider()
st.caption(CONFIG["disclaimer"])


# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-08)  New file. Added: CONFIG block, FIT, load_table (cached call
#   to file_io.read_table), cached_prices, guess (= file_io.guess_column), pick,
#   as_text_dates, style_metrics, wealth_line, the upload and column-mapping
#   steps, the sidebar settings, and the Summary, Charts, Strategy detail and
#   Data checks tabs.
