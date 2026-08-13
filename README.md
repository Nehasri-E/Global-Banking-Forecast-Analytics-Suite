# Global Banking Forecast Analytics Suite

A forecasting and risk analytics platform for six major banking stocks across the US, Germany, Japan, and India, comparing statistical and deep learning models to see which actually performs best for bank stock price prediction.

I built this to test a real question: do deep learning models actually beat classical statistical methods for forecasting bank stocks, or is that assumption overrated? I picked JPMorgan, Citigroup, Deutsche Bank, Nomura, HDFC Bank, and Axis Bank to get a genuine cross-country comparison like US, European, Japanese, and Indian banking stocks all behave differently, so a model that wins for one market isn't guaranteed to win for another.

## Results

Metrics below are averaged across all six stocks, from walk-forward validation (not a single train/test split that is why the model is retrained and re-tested as it moves forward in time, closer to how forecasting actually works in practice).

| Model | Avg MAE | Avg RMSE |
|---|---|---|
| ARIMA | 2.33 | 3.48 |
| Hybrid ARIMA-LSTM | 4.18 | 5.22 |
| LSTM | 7.05 | 8.44 |

**ARIMA won outright — and by a wide margin.** This was genuinely surprising going in, since LSTM is the "fancier" model. But bank stock prices over short horizons behave close to a random walk, and ARIMA's simpler linear structure handles that better than LSTM, which needs more data and tends to overfit noise on daily-level financial series. The Hybrid model landed in between, which makes sense since it blends both.

For each stock, the pattern held consistently, ARIMA had the lowest error for every single one of the six banks, from Deutsche Bank (ARIMA MAE 0.15) to Axis Bank (ARIMA MAE 10.72), which had the highest absolute error of the group, likely due to higher volatility in Indian banking stocks over the period analyzed.

## How it works

Ten years of daily OHLCV data is pulled per stock via the yfinance API. Since raw price series aren't stationary, I applied first-order differencing and ran ADF tests to confirm stationarity before feeding data into ARIMA.

Three forecasting approaches are run and compared:
- **ARIMA** — classical statistical time-series model, the eventual winner
- **LSTM** — a recurrent neural network built to capture sequential/nonlinear patterns
- **Hybrid ARIMA-LSTM** — combines both model outputs, aiming to capture linear trend (ARIMA) and nonlinear residual patterns (LSTM) together

All three are evaluated using **walk-forward validation** rather than a static train/test split, since that better reflects how forecasting is actually used in practice — the model doesn't get to see the future in one shot, it's continuously re-tested as time moves forward.

Beyond forecasting, the platform layers on risk analytics: annualized volatility, Sharpe Ratio, and 95% forecast confidence intervals, then ranks all six banks by risk-adjusted performance on an interactive Streamlit dashboard.

## Limitations

Results reflect the specific historical window used for training and testing, so performance could shift in a different market regime (e.g. a high-volatility crisis period). LSTM's underperformance here is somewhat specific to short daily-horizon forecasting on relatively small per-stock datasets. It may perform differently with more data or longer forecast horizons. The Hybrid model is a simple output-blend rather than a jointly-trained architecture, so there's room to make that combination smarter.

## Tech stack

Python, Pandas, NumPy, Statsmodels (ARIMA), TensorFlow/Keras (LSTM), Streamlit, Matplotlib

## Running it

```
git clone https://github.com/Nehasri-E/Global-Banking-Forecast-Analytics-Suite.git
cd Global-Banking-Forecast-Analytics-Suite
pip install -r requirements.txt
python src/run_pipeline.py
streamlit run app.py
```

## What I'd build next

Value-at-Risk (VaR) analysis and an efficient frontier view to turn this from single-stock forecasting into actual portfolio-level decision support.

## Screenshots

![Executive Dashboard](assets/dashboard_home.png)
![Forecast Comparison](assets/forecast_comparison.png)
![Global Banking Ranking](assets/global_ranking.png)

## Author

Nehasri Eragandula — IIT Kanpur, Biological Sciences and Bioengineering