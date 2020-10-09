# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:47:10 2019
Dataquest Step6 Course6: Decision Tree
Mission5: Guided Project - Predicting Bike Rentals
"""


import pandas as pd
import matplotlib.pyplot as plt


bike_rentals = pd.read_csv('bike_rentals.csv')
print(bike_rentals.head())

plt.figure(1)
plt.hist(bike_rentals['cnt'])

bike_rentals.corr()["cnt"]



def assign_label(hour):
    if hour >= 6 and hour < 12:
        return 1
    elif hour >= 12 and hour < 18:
        return 2
    elif hour >= 18 and hour < 24:
        return 3
    elif hour >= 0 and hour < 6:
        return 4
    
bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)



train = bike_rentals.sample(frac=0.8)
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]

cols = list(bike_rentals.columns)
cols.remove('dteday')
cols.remove('casual')
cols.remove('registered')
cols.remove("cnt")



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(train[cols],train['cnt'])
predictions = lr.predict(test[cols])
mse = mean_squared_error(test['cnt'], predictions)
print('mse: ', mse)



from sklearn.tree import DecisionTreeRegressor 

dtr = DecisionTreeRegressor(min_samples_leaf=5)
dtr.fit(train[cols],train['cnt'])
predictions_dtr = dtr.predict(test[cols])
mse_dtr = mean_squared_error(test['cnt'], predictions_dtr)
print('Mean square error for decision tree regressor: ', mse_dtr)

dtr = DecisionTreeRegressor(min_samples_leaf=5, max_depth=14)
dtr.fit(train[cols],train['cnt'])
predictions_dtr = dtr.predict(test[cols])
mse_dtr = mean_squared_error(test['cnt'], predictions_dtr)
print('Mean square error for decision tree regressor: ', mse_dtr)



from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(train[cols],train['cnt'])
predictions_rfr = rfr.predict(test[cols])
mse_rfr = mean_squared_error(test['cnt'], predictions_rfr)
print('Mean square error for random forest regressor (min sample leafs = 5, max depth = 14): ', mse_rfr)


rfr = RandomForestRegressor(min_samples_leaf=5, max_depth=14)
rfr.fit(train[cols],train['cnt'])
predictions_rfr = rfr.predict(test[cols])
mse_rfr = mean_squared_error(test['cnt'], predictions_rfr)
print('Mean square error for random forest regressor (min sample leafs = 5, max depth = 14): ', mse_rfr)


comparison = test['cnt']
comparison = comparison.reset_index()

plt.plot(comparison['cnt'],'b.')
plt.plot(predictions_rfr, 'r.')
plt.show()


