# Nexuvia Streamlit tools

Two separate Streamlit apps that share one repository.

| App | Main file | Calculations |
| --- | --- | --- |
| Out-of-sample performance test | `oos_app.py` | `oos_metrics.py` |
| Technical analysis (RSI, Connors RSI, MACD, Williams %R) | `ta_app.py` | `ta_indicators.py` |

`file_io.py` reads the uploaded files for both apps. It detects the delimiter (comma, semicolon, tab) and the decimal mark (point or comma) separately for every file. The calculation modules contain no Streamlit code and can be imported in Spyder.

## Run locally

From an Anaconda prompt in this folder:

```
pip install -r requirements.txt
streamlit run oos_app.py
streamlit run ta_app.py
```

## Deploy from GitHub

Push the folder to a GitHub repository. In Streamlit Community Cloud, create one app per main file (`oos_app.py` and `ta_app.py`), each pointing to the same repository and branch. The `.gitignore` excludes data files; do not commit weight files, price files or optimizer code to a public repository.

## Out-of-sample test: how it works

Upload one or more weight files (date, strategy, asset, weight) and one price file. Column names are detected automatically and can be changed in the mapping boxes.

- Weights are rescaled to sum to 1 per date and strategy. Absolute weights up to the zero tolerance (default 1e-12) are set to zero, which removes numerical residuals such as -1e-17.
- The last price date on or before the recommendation date is the signal day. The portfolio is bought at the close of the signal day plus the purchase lag (default 1 trading day). Returns are earned from the following price date.
- A holding period ends at the purchase close of the next recommendation, or at the last price date. With "buy and hold" the weights drift with prices; with "constant mix" they are reset every period.
- Weight on assets without a price is kept as cash with 0 % return, or spread over the matched assets. The share of matched weight is shown for every period.
- Transaction cost is the entered basis points times the traded value at each purchase.
- If the weight file has an estimation end date, the page reports whether the evaluated returns start after it.
- Statistics: cumulative and annualized return, volatility, Sharpe, Sortino, Omega, maximum drawdown, Calmar, historical VaR and CVaR at 95 %, and with a benchmark column: tracking error, information ratio, beta, Jensen's alpha and Treynor ratio. Excess returns use the risk-free rate entered in the sidebar.
- Prices must be in one currency. The page does not convert currencies and does not model dividends unless the prices are adjusted.

## Technical analysis: conventions

- RSI uses Wilder's smoothing, seeded with the simple mean of the first `length` price changes.
- MACD uses EMAs seeded with the simple mean of the first `span` values: line = fast EMA minus slow EMA, signal = EMA of the line, histogram = line minus signal.
- Connors RSI is the mean of RSI(close, 3), RSI(streak, 2) and the percent rank of the one-period return over 100 periods. Percent rank is the share of the previous 100 returns that are strictly lower than the current one. Other software may treat ties differently.
- Williams %R = -100 × (highest high − close) / (highest high − lowest low) over `length` periods.
- Indicators are computed on the full uploaded history, and the date range in the app only changes what is displayed.
- The default overbought and oversold levels (RSI 70/30, Connors RSI 90/10, Williams %R −20/−80) are adjustable in the sidebar.

Results of both apps are calculations on uploaded data. They are not investment advice.
