#Problem I. Consider the monthly ice cream production data from the year 2010 on-
#ward.
#1. Construct a time plot of the ice cream production data from the year 2010 onward. Describe any visible patterns such as trend, seasonality, or irregularity.
#2. Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the series up to lag 12. Based on the plots, comment on the possible dependence structure of the data.
#3. Perform the Augmented Dickey–Fuller (ADF) test on the series to assess stationarity. Report the test statistic, p-value, and conclude whether the series is stationary or requires differencing.

import pandas as pd
import matplotlib.pyplot as plt

#1.loading dataset
df = pd.read_csv("ice_cream.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df_2010 = df.loc['2010-01-01':]

#Plot
plt.figure(figsize=(10,4))
df_2010['IPN31152N'].plot()
plt.title("Ice Cream production from the year 2010 onward")
plt.xlabel("Year")
plt.ylabel("Production")
plt.show()

#2.ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize =(12,4))
plot_acf(df_2010['IPN31152N'], lags=12, ax=axes[0])
axes[0].set_title("ACF(lag 12)")

plot_pacf(df_2010['IPN31152N'], lags=12, ax=axes[1])
axes[1].set_title("PACF(lag 12)")
plt.tight_layout()
plt.show()

#3 ADF
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df_2010['IPN31152N'])
print("ADF Test results:")
print(f"Test Statistics:{adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"{key}:{value}")

#Problem 2
#1 Plot and ACF and PACF
cf = pd.read_csv("catfish.csv")
cf['Date'] = pd.to_datetime(cf['Date'])
cf.set_index('Date', inplace=True)


plt.figure(figsize=(10,4))
cf['Total'].plot()
plt.title("Catfish sales")
plt.xlabel("Year")
plt.ylabel("sales")
plt.grid(True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize =(12,4))
plot_acf(cf['Total'], lags=24, ax=axes[0])
axes[0].set_title("ACF(lag 24)")

plot_pacf(cf['Total'], lags=24, ax=axes[1])
axes[1].set_title("PACF(lag 24)")
plt.tight_layout()
plt.show()

#2 ARMA
from statsmodels.tsa.arima.model import ARIMA
cf_train = cf.loc['2000-01-01':'2010-12-01']
cf_test = cf.loc['2011-01-01':'2012-12-01']

p=2
q=1
model = ARIMA(cf_train['Total'], order=(p, 1, q))
fitted = model.fit()

print (fitted.summary())

#3 RMSE
import numpy as np

forecast = fitted.forecast(steps=len(cf_test))
cf_test['forecast'] = forecast.values
rmse = np.sqrt(np.mean((cf_test['Total']) - cf_test['forecast'])**2)
print("RMSE=", rmse)

#4
future = fitted.get_forecast(steps=12)
mean_forecast = future.predicted_mean
confidence_intervals = future.conf_int()

plt.figure(figsize=(10,4))
plt.plot(cf_train['Total'], label="historical")
plt.plot(mean_forecast.index, mean_forecast, label="Forecast (2013)")

plt.fill_between(
    mean_forecast.index,
    confidence_intervals.iloc[:,0],
    confidence_intervals.iloc[:,1],
    alpha=0.3
)
plt.title("Catfish Sales Forecast (2013) with 95% CI")
plt.legend()
plt.show()