# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:56:13 2019
Dataquest Step6 Course5: Logistic Regression
Mission7: Guided Project: Predicting Stock Market
"""

import matplotlib.pyplot as plt
import yfinance as yf
import datetime 


first_jan = '1971-01-01'
today = datetime.datetime.now()

# load SP500 historical data from yahoo finance
data = yf.download("^GSPC", start=first_jan, end=today)
data['DateTime'] = data.index


# Calculate weekly, monthly and yearly rolling average and weekly to yearly ratio and plot
data['5day mean'] = data['Close'].rolling(5).mean().shift(1)
data['month mean'] = data['Close'].rolling(22).mean().shift(1)
data['year mean'] = data['Close'].rolling(260).mean().shift(1)
data['ratio mean'] = data['5day mean']/data['year mean']

plt.figure(1)
plt.plot_date(data['DateTime'], data['Close'], c='b')
plt.plot_date(data['DateTime'], data['month mean'], c='g', ls='-')
plt.plot_date(data['DateTime'], data['year mean'], c='r', ls='-')
plt.title('S&P500 with monthly and yearly rolling average')
plt.show()


# calculate standard deviations and plot
data['5day std'] = data['Close'].rolling(5).std().shift(1)
data['month std'] = data['Close'].rolling(22).std().shift(1)
data['year std'] = data['Close'].rolling(260).std().shift(1)
data['ratio std'] = data['5day std']/data['year std']

plt.figure(2)
plt.plot_date(data['DateTime'],data['year std'],c='r')
plt.plot_date(data['DateTime'],data['month std'],c='g')
plt.plot_date(data['DateTime'],data['5day std'],c='b')
plt.title('standard deviations of S&P500 prices on weekly, monthly and yearly basis')
plt.show()


# drop all rows without yearly rolling average
data = data.dropna(axis=0)

# split into train (47 years) and test set (~3 years)
train = data[data["DateTime"] <= datetime.datetime(year=2018, month=1, day=1)]
test = data[data["DateTime"] > datetime.datetime(year=2018, month=1, day=1)]


# use linear regression for prediction with rolling means and standard deviations as features
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

features = ['5day mean','month mean','year mean','ratio mean',
            '5day std','month std','year std','ratio std']
model = LinearRegression()
model.fit(train[features],train['Close'])
predictions = model.predict(test[features])
mae = mean_absolute_error(predictions, test['Close'])


# plot actual close prices together with predicted close prices
plt.figure(3)
plt.plot_date(test['DateTime'],test['Close'],c='b')
plt.plot_date(test['DateTime'],predictions,c='g')
plt.show()



