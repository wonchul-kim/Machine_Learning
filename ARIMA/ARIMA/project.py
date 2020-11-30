
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt

import math

sns.set_style('darkgrid')

data_filename = './000660.KS.csv' 
pd_data = pd.read_csv(data_filename).fillna(0) 
pd_data = pd_data[pd_data['Close'] != 0]


prob_dist_close = pd_data['Close']

res = seasonal_decompose(prob_dist_close, model='multiplicative', freq = 30)

prob_dist_close_log = np.log(prob_dist_close)

train_data, test_data = prob_dist_close_log[3:int(len(prob_dist_close_log)*0.9)], \
            prob_dist_close_log[int(len(prob_dist_close_log)*0.9):]

model = ARIMA(train_data, order=(1, 1, 2))  
fitted_m = model.fit(disp=-1)  

fc, se, conf = fitted_m.forecast(len(test_data), alpha=0.05)  

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, c='b', label='actual price')
plt.plot(fc_series, c='r',label='predicted price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.legend()
plt.show()


mse = mean_squared_error(test_data, fc)
print('MSE: ', mse)

mae = mean_absolute_error(test_data, fc)
print('MAE: ', mae)

rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: ', rmse)

mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: {:.2f}%'.format(mape*100))




