import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("catfish.csv")

# Detect date column
date_col = df.columns[0]
for c in df.columns:
    if "date" in c.lower() or "month" in c.lower():
        date_col = c
        break

# Detect sales column
sales_col = df.columns[1]
for c in df.columns:
    if "sale" in c.lower() or "value" in c.lower():
        sales_col = c
        break

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).set_index(date_col)

y = df[sales_col].astype(float).asfreq("MS").fillna(method="ffill")


# 2. Train-test split
train = y.loc['2000-01-01':'2010-12-01']
test = y.loc['2011-01-01':'2012-12-01']
popa = y.loc['2000-01-01':'2012-12-01']

# 3. SARIMA candidate models

candidates = [
    ((3,1,2), (3,1,2,12)),
    ((2,1,2), (2,1,2,12)),
    ((4,1,2), (4,1,2,12))
]

results = []

for order, seas in candidates:
    print(f"\nFitting SARIMA {order} x {seas}")
    model = SARIMAX(train, order=order, seasonal_order=seas,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)
    print(f" -> AIC={res.aic:.3f}, BIC={res.bic:.3f}")
    results.append((order, seas, res.aic, res.bic, res))


# 4. Select best model by AIC/BIC

best = min(results, key=lambda x: x[2])  # AIC winner
order, seas, aic, bic, chosen_model = best

print("\nSelected Model:")
print("Order =", order)
print("Seasonal =", seas)
print("AIC =", aic)
print("BIC =", bic)


# 5. Forecast & compute RMSE
test_periods = 24
forecast = chosen_model.get_forecast(steps=test_periods)
pred = forecast.predicted_mean
pred.index = test.index

rmse = sqrt(mean_squared_error(test, pred))
print("\nRMSE =", rmse)


# 6. Plot actual vs predicted

plt.figure(figsize=(10,5))
plt.plot(popa, label="Actual")
plt.plot(pred, label="Forecast", linestyle="--")
plt.axvline(train.index[-1], color="k", linestyle=":")
plt.title(f"SARIMA {order} x {seas} Forecast\nRMSE = {rmse:.3f}")
plt.legend()
plt.tight_layout()
plt.show()

print("\nLast 10 Actual vs Predicted:")
print(pd.DataFrame({"Actual": test, "Predicted": pred}).tail(10))
