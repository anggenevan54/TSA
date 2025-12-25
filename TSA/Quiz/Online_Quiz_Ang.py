import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX # Added from second script


# User-adjustable filename for Catfish data
FNAME = "catfish.csv"


if not os.path.exists(FNAME):
    raise FileNotFoundError(f"File '{FNAME}' not found in working directory: {os.listdir('.')}")

# 1) Read file and identify columns
df = pd.read_csv(FNAME)

# Try to find a date column
date_col = None
for c in df.columns:
    if 'date' in c.lower() or 'year' in c.lower() or 'month' in c.lower():
        date_col = c
        break

if date_col is None:
    date_col = df.columns[0]  # fallback to first column

# Handle case where there are separate year and month columns
cols_lower = [c.lower() for c in df.columns]
if 'year' in cols_lower and 'month' in cols_lower:
    yc = df.columns[cols_lower.index('year')]
    mc = df.columns[cols_lower.index('month')]
    df['date'] = pd.to_datetime(df[yc].astype(int).astype(str) + '-' + df[mc].astype(int).astype(str) + '-01')
    date_col = 'date'
else:
    # try parsing existing date column
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        # if it's numeric year only
        if pd.api.types.is_numeric_dtype(df[date_col]):
            df['date'] = pd.to_datetime(df[date_col].astype(int).astype(str) + '-01-01')
            date_col = 'date'
        else:
            raise ValueError("Could not parse the date column. Please ensure file has a parseable date column.")

# Choose sales column: prefer names with sale/sales/catfish
value_col = None
for c in df.columns:
    if c == date_col:
        continue
    if any(k in c.lower() for k in ['sale','sales','catfish','value','quantity']):
        value_col = c
        break

if value_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric column found for sales. Please provide a numeric sales column.")
    value_col = numeric_cols[0]

# Prepare time series (monthly)
df = df[[date_col, value_col]].dropna().copy()
df.columns = ['date', 'sales']
df = df.sort_values('date').reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').asfreq('MS')  # monthly start frequency

# Subset to 2000-01 - 2010-12 as requested (if data available)
start_req = pd.to_datetime('2000-01-01')
end_req = pd.to_datetime('2010-12-01')
if (df.index.min() <= start_req) and (df.index.max() >= end_req):
    ts = df.loc[start_req:end_req, 'sales'].copy()
else:
    # if range not fully available, try to subset overlapping portion
    ts = df['sales'].copy()
    ts = ts[(ts.index >= start_req) & (ts.index <= end_req)]
    if len(ts) == 0:
        raise ValueError("No data available in the 2000-01 to 2010-12 window. Please check your CSV.")

print("Time series period:", ts.index.min().date(), "to", ts.index.max().date(), " (n=", len(ts), ")")


# Part (a) ACF/PACF up to lag 20
plt.figure(figsize=(10,3))
plt.plot(ts); plt.title('Monthly Catfish Sales'); plt.ylabel('Sales'); plt.xlabel('Date')
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(2,1, figsize=(10,6))
plot_acf(ts.dropna(), lags=20, ax=axes[0]); axes[0].set_title('ACF (lags up to 20)')
plot_pacf(ts.dropna(), lags=20, ax=axes[1], method='ywm'); axes[1].set_title('PACF (lags up to 20)')
plt.tight_layout(); plt.show()

print("Note: Look for slow decay in ACF (nonstationarity) and spikes in PACF suggesting AR order; check seasonal lag 12 for monthly data.")

# Seasonality check
decomp = seasonal_decompose(ts.dropna(), model='additive', period=12, extrapolate_trend='freq')
fig = decomp.plot()
fig.set_size_inches(10,8)
plt.suptitle('Seasonal Decomposition (period=12)'); plt.show()

# ADF test to assess d
adf_stat, adf_p, *_ = adfuller(ts.dropna())
print(f"ADF statistic = {adf_stat:.4f}, p-value = {adf_p:.4f}")
d = 0 if adf_p < 0.05 else 1
print("Choosing d =", d)

# Part (b) Split: train 2000-01 to 2009-12, test 2010-01 to 2010-12
train = ts['2000-01-01':'2009-12-01'].copy()
test = ts['2010-01-01':'2010-12-01'].copy()
if len(train)==0 or len(test)==0:
    raise ValueError("Train or test segment empty. Ensure data covers 2000-2010.")
print("Training:", train.index.min().date(), "to", train.index.max().date(), " (n=",len(train),")")
print("Testing:", test.index.min().date(), "to", test.index.max().date(), " (n=",len(test),")")

# Utility: grid-search for best (p,d,q) by AIC (p,q up to 4)
def select_order_by_aic(series, p_max=4, q_max=4, d=0):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in range(p_max+1):
        for q in range(q_max+1):
            try:
                m = ARIMA(series, order=(p,d,q))
                res = m.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p,d,q)
                    best_model = res
            except Exception:
                continue
    return best_order, best_model

print("Selecting ARIMA(p,d,q) on training by AIC (p,q <= 4)...")
best_order_arima, best_model_arima = select_order_by_aic(train, p_max=4, q_max=4, d=d)
print("Best ARIMA order:", best_order_arima)

# Residual diagnostics for ARIMA
resid = best_model_arima.resid
plt.figure(figsize=(10,3)); plt.plot(resid); plt.title(f'ARIMA{best_order_arima} Residuals'); plt.show()
plt.figure(figsize=(10,3)); plot_acf(resid.dropna(), lags=20); plt.title('ACF of ARIMA residuals'); plt.show()
lb = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
print("Ljung-Box (lag=10):\n", lb)

# Conclusion on adequacy
if lb['lb_pvalue'].iloc[0] > 0.05:
    print("ARIMA residuals appear uncorrelated by Ljung-Box (good).")
else:
    print("ARIMA residuals show significant autocorrelation by Ljung-Box (may be inadequate).")


# Part (d) Fit ARMA (force d=0)
print("Selecting ARMA(p,q) on training by AIC (forcing d=0)...")
best_order_arma, best_model_arma = select_order_by_aic(train, p_max=4, q_max=4, d=0)
print("Best ARMA order (ARIMA(p,0,q)):", best_order_arma)

resid_arma = best_model_arma.resid
plt.figure(figsize=(10,3)); plt.plot(resid_arma); plt.title(f'ARMA{best_order_arma} Residuals'); plt.show()
plt.figure(figsize=(10,3)); plot_acf(resid_arma.dropna(), lags=20); plt.title('ACF of ARMA residuals'); plt.show()
lb_arma = acorr_ljungbox(resid_arma.dropna(), lags=[10], return_df=True)
print("Ljung-Box (lag=10) for ARMA:\n", lb_arma)
if lb_arma['lb_pvalue'].iloc[0] > 0.05:
    print("ARMA residuals appear uncorrelated by Ljung-Box (good).")
else:
    print("ARMA residuals show significant autocorrelation by Ljung-Box (may be inadequate).")

# Part (e) Rolling one-step-ahead forecasts on test (refit each step)
def rolling_refit_forecast(order, train_series, test_series):
    history = train_series.copy()
    preds = []
    for t in range(len(test_series)):
        model = ARIMA(history, order=order).fit()
        fc = model.get_forecast(steps=1)
        yhat = float(fc.predicted_mean.iloc[-1])
        preds.append(yhat)
        # update history by appending actual value
        history = pd.concat([history, pd.Series([test_series.iloc[t]], index=[test_series.index[t]])])
    return np.array(preds)

print("Generating rolling one-step-ahead forecasts (refitting at each step)...")
preds_arima = rolling_refit_forecast(best_order_arima, train, test)
preds_arma  = rolling_refit_forecast(best_order_arma,  train, test)

rmse_arima = math.sqrt(mean_squared_error(test.values, preds_arima))
rmse_arma  = math.sqrt(mean_squared_error(test.values, preds_arma))
print(f"Test RMSE -> ARIMA{best_order_arima}: {rmse_arima:.4f}; ARMA{best_order_arma}: {rmse_arma:.4f}")

if rmse_arima < rmse_arma:
    winner = ('ARIMA', best_order_arima)
else:
    winner = ('ARMA', best_order_arma)
print("Better model on test set:", winner)

print("Reminder: Use testing set only for evaluating forecast accuracy to avoid optimistic (overfit) estimates from training data.")

# Plot actual vs forecasts
plt.figure(figsize=(10,4))
plt.plot(train.index, train.values, label='Train')
plt.plot(test.index, test.values, label='Test (actual)')
plt.plot(test.index, preds_arima, label=f'ARIMA{best_order_arima} preds', marker='o')
plt.plot(test.index, preds_arma, label=f'ARMA{best_order_arma} preds', marker='x')
plt.legend(); plt.title('One-step-ahead rolling forecasts on test period'); plt.show()


# Part (f) Refit winner to full 2000-01..2010-12 and forecast Jan 2011-Dec 2011
full = ts.copy()
best_model_name = winner[0]
best_model_order = winner[1]
print("Refitting best model to full data:", best_model_name, best_model_order)
best_full = ARIMA(full, order=best_model_order).fit()
fc = best_full.get_forecast(steps=12, alpha=0.05)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)
fc_index = pd.date_range(start=full.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

fc_mean.index = fc_index
fc_ci.index = fc_index

plt.figure(figsize=(10,4))
plt.plot(full, label='Observed')
plt.plot(fc_mean, label='Forecast (2011)')
plt.fill_between(fc_index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.25, label='95% PI')
plt.legend(); plt.title(f'Forecasts Jan 2011 - Dec 2011 using {best_model_name}{best_model_order}'); plt.show()

# Save outputs
os.makedirs('catfish_outputs', exist_ok=True)
with open('catfish_outputs/summary.txt', 'w') as f:
    f.write(f"Best ARIMA order (training by AIC): {best_order_arima}\n")
    f.write(f"Best ARMA order (training by AIC): {best_order_arma}\n")
    f.write(f"Test RMSE -> ARIMA: {rmse_arima:.6f}, ARMA: {rmse_arma:.6f}\n")
    f.write(f"Winner: {winner}\n")
print("Saved summary to catfish_outputs/summary.txt")



# User-adjustable variables for Electric data
ELECTRIC_INPUT_CSV = "electricData.csv"
ELECTRIC_OUTPUT_CSV = "electric_forecasts_2017_results.csv"
ELECTRIC_TRAIN_START = "1986-01-01"
ELECTRIC_TRAIN_END   = "2016-12-31"
ELECTRIC_TEST_START  = "2017-01-01"
ELECTRIC_TEST_END    = "2017-12-31"
SARIMA_ORDER = (1, 1, 0)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 12)

def safe_read_csv(path):
    electric_df_local = pd.read_csv(path)
    return electric_df_local

def parse_time_index(electric_df_local):
    """
    Attempt to parse a date/time index for monthly data.
    Tries common patterns, falls back to generating a monthly index if obvious.
    """
    # Look for common date columns
    date_candidates = [c for c in electric_df_local.columns if 'date' in c.lower() or 'month' in c.lower() or 'year' in c.lower()]
    parsed = None
    if date_candidates:
        # Try first candidate
        for c in date_candidates:
            parsed = pd.to_datetime(electric_df_local[c], errors='coerce')
            if parsed.notna().sum() > 0:
                electric_df_local['_parsed_date'] = parsed
                break
    # If nothing found, try the first column
    if '_parsed_date' not in electric_df_local.columns:
        first = electric_df_local.columns[0]
        parsed = pd.to_datetime(electric_df_local[first], errors='coerce')
        electric_df_local['_parsed_date'] = parsed

    # If parsed are mostly NaT, try to construct index assuming the file rows are monthly observations
    if electric_df_local['_parsed_date'].isna().mean() > 0.5:
        # Attempt: if there's a Year and Month columns use them
        year_col_local = None
        month_col_local = None
        for c in electric_df_local.columns:
            if c.lower().startswith('year'):
                year_col_local = c
            if c.lower().startswith('month'):
                month_col_local = c
        if year_col_local and month_col_local:
            electric_df_local['_parsed_date'] = pd.to_datetime(electric_df_local[year_col_local].astype(int).astype(str) + '-' +
                                                        electric_df_local[month_col_local].astype(int).astype(str).str.zfill(2) + '-01')
        else:
            # But we will try to detect if a string like "1985-01" appears in any cell
            possible_col_local = None
            for c in electric_df_local.columns:
                if electric_df_local[c].astype(str).str.contains(r'^\d{4}-\d{2}').any():
                    possible_col_local = c
                    break
            if possible_col_local is not None:
                electric_df_local['_parsed_date'] = pd.to_datetime(electric_df_local[possible_col_local].astype(str).str.extract(r'(^\d{4}-\d{2})')[0] + '-01',
                                                            errors='coerce')
            else:
                # final fallback: create a monthly index starting Jan 1985 and use length of df
                start_local = pd.to_datetime("1985-01-01")
                electric_df_local['_parsed_date'] = pd.date_range(start=start_local, periods=len(electric_df_local), freq='MS')

    # set index and return
    electric_df_local = electric_df_local.set_index('_parsed_date').sort_index()
    return electric_df_local

def detect_value_column(electric_df_local):
    # pick first numeric column (not the index)
    for c in electric_df_local.columns:
        if pd.api.types.is_numeric_dtype(electric_df_local[c]) and c != '_parsed_date':
            return c
    # else try common names
    for name_local in ['Value','value','Electricity','Consumption','consumption']:
        if name_local in electric_df_local.columns:
            return name_local
    # fallback to second column if first is date
    cols_local = list(electric_df_local.columns)
    if len(cols_local) >= 1:
        return cols_local[0]
    raise ValueError("Could not detect value column in CSV. Ensure it contains a numeric column with the series.")

def fill_missing(electric_series_local):
    # If there are NaNs, use time interpolation to keep monthly continuity
    n_missing_local = electric_series_local.isna().sum()
    if n_missing_local > 0:
        print(f"Found {n_missing_local} missing values in the series. Applying time-based interpolation.")
        electric_series_local = electric_series_local.interpolate(method='time')
        remaining_local = electric_series_local.isna().sum()
        if remaining_local > 0:
            print(f"After interpolation, {remaining_local} missing remain. Filling forward then backward.")
            electric_series_local = electric_series_local.fillna(method='ffill').fillna(method='bfill')
    return electric_series_local

def fit_sarima(y_local, order_local, seasonal_order_local):
    model_local = SARIMAX(y_local, order=order_local, seasonal_order=seasonal_order_local,
                        enforce_stationarity=False, enforce_invertibility=False)
    try:
        sarima_results_local = model_local.fit(disp=False)
        return sarima_results_local
    except Exception as e_local:
        # Try alternative optimizers
        for method_local in ['powell', 'lbfgs', 'bfgs', 'nm']:
            try:
                sarima_results_local = model_local.fit(method=method_local, maxiter=200, disp=False)
                print(f"Fit succeeded using method={method_local}")
                return sarima_results_local
            except Exception:
                continue
        # final attempt (let statsmodels decide)
        sarima_results_local = model_local.fit(disp=False)
        return sarima_results_local

def compute_mape(true_local, pred_local):
    true_local = np.array(true_local); pred_local = np.array(pred_local)
    mask_local = true_local != 0
    if mask_local.sum() == 0:
        return np.nan
    return np.mean(np.abs((true_local[mask_local] - pred_local[mask_local]) / true_local[mask_local])) * 100.0


# Main execution for Electric data analysis

if __name__ == "__main__":
    if not os.path.exists(ELECTRIC_INPUT_CSV):
        raise FileNotFoundError(f"{ELECTRIC_INPUT_CSV} not found. Place the file in the same folder as this script.")

    electric_df_raw = safe_read_csv(ELECTRIC_INPUT_CSV)
    print("Columns detected in CSV:", electric_df_raw.columns.tolist())

    electric_df_processed = parse_time_index(electric_df_raw.copy())
    # detect value column
    electric_value_col = detect_value_column(electric_df_processed)
    print("Using value column:", electric_value_col)

    electric_series = electric_df_processed[electric_value_col].astype(float)
    # force monthly start-of-month frequency for consistency
    electric_series.index = pd.to_datetime(electric_series.index).to_period('M').to_timestamp('M')  # end-of-month index
    electric_series = electric_series.asfreq('M')

    print("Series range:", electric_series.index.min().date(), "to", electric_series.index.max().date())
    print("Total observations:", len(electric_series))
    print("Missing values before fill:", electric_series.isna().sum())

    electric_series = fill_missing(electric_series)
    print("Missing after fill:", electric_series.isna().sum())

    # enforce user-specified training/testing periods
    electric_train = electric_series[ELECTRIC_TRAIN_START:ELECTRIC_TRAIN_END]
    electric_test = electric_series[ELECTRIC_TEST_START:ELECTRIC_TEST_END]

    print(f"Training observations: {len(electric_train)} ({electric_train.index.min().date()} to {electric_train.index.max().date()})")
    print(f"Testing observations: {len(electric_test)} ({electric_test.index.min().date()} to {electric_test.index.max().date()})")

    # check length
    if len(electric_train) < 24:
        raise ValueError("Training period too short for SARIMA seasonal modeling - need more observations.")

    # Part (a): Fit on training
    print("\nFitting SARIMA{} x {} on training data...".format(SARIMA_ORDER, SARIMA_SEASONAL_ORDER))
    sarima_results = fit_sarima(electric_train, SARIMA_ORDER, SARIMA_SEASONAL_ORDER)
    print("\nModel summary:")
    print(sarima_results.summary())

    # Residual diagnostics
    sarima_resid = sarima_results.resid.dropna()
    print("\nResiduals count:", len(sarima_resid))
    max_lag_local = min(24, len(sarima_resid)-1)
    if max_lag_local >= 12:
        lb_local = acorr_ljungbox(sarima_resid, lags=[12, max_lag_local], return_df=True)
        print("\nLjung-Box test (lags 12 and {0}):".format(max_lag_local))
        print(lb_local)
    else:
        print("Not enough residuals to run Ljung-Box with lag 12/24; skipping.")

    # Quick residual plots
    plt.figure(figsize=(10,3))
    plt.plot(sarima_resid)
    plt.title("Residuals from fitted SARIMA model (training)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,3))
    plt.hist(sarima_resid, bins=20)
    plt.title("Histogram of residuals")
    plt.tight_layout()
    plt.show()

    # Part (b): Rolling one-step-ahead forecasts for each month in 2017
    electric_history = electric_train.copy()
    electric_predictions = []
    electric_pred_index = []

    print("\nPerforming rolling one-step-ahead forecasts for each month of 2017 (re-fitting each time)...")
    for idx_local in electric_test.index:
        # fit on available history
        r_sarima_local = fit_sarima(electric_history, SARIMA_ORDER, SARIMA_SEASONAL_ORDER)
        fc_local = r_sarima_local.get_forecast(steps=1)
        pred_local = float(fc_local.predicted_mean.iloc[0])
        electric_predictions.append(pred_local)
        electric_pred_index.append(idx_local)
        # append actual observed value from test to history (recursive update)
        electric_history = pd.concat([electric_history, pd.Series({idx_local: float(electric_test.loc[idx_local])})])

    electric_preds = pd.Series(electric_predictions, index=electric_pred_index, name='forecast_2017')

    # Part (c): Forecast accuracy
    electric_test_aligned = electric_test.reindex(electric_preds.index)
    mape_val = compute_mape(electric_test_aligned.values, electric_preds.values)
    rmse_val = np.sqrt(mean_squared_error(electric_test_aligned.values, electric_preds.values))

    print("\nForecast accuracy for 2017:")
    print(f"  MAPE = {mape_val:.4f}%")
    print(f"  RMSE = {rmse_val:.4f} (same units as series)")

    # Save results
    electric_forecast_results = pd.DataFrame({
            'actual_2017': electric_test_aligned,
            'forecast_2017': electric_preds,
            'error': electric_test_aligned - electric_preds
    })
    electric_forecast_results.to_csv(ELECTRIC_OUTPUT_CSV)
    print("\nSaved forecast results to:", ELECTRIC_OUTPUT_CSV)

    # Plot observed vs forecasts
    plt.figure(figsize=(12,4))
    plt.plot(electric_series.index, electric_series.values, label='Observed (all)')
    plt.plot(electric_preds.index, electric_preds.values, linestyle='--', marker='o', label='Rolling 1-step Forecast (2017)')
    plt.plot(electric_test.index, electric_test.values, linestyle='-', marker='x', label='Actual 2017')
    plt.legend()
    plt.title('Monthly Electric Consumption: Observed vs Rolling 1-step Forecasts (2017)')
    plt.xlabel('Date')
    plt.ylabel(electric_value_col)
    plt.tight_layout()
    plt.show()

    # Plot forecast errors
    plt.figure(figsize=(10,3))
    plt.plot(electric_forecast_results.index, electric_forecast_results['error'].values, marker='o')
    plt.axhline(0, linestyle='--')
    plt.title('Forecast Errors (Actual - Forecast) for 2017')
    plt.tight_layout()
    plt.show()