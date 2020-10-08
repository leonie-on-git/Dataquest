# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:34:02 2019
Dataquest Step6 Course1: Machine Learning Fundamentals
Mission6: Guided Project - Predicting Car Prices
"""

import pandas as pd
import numpy as np


# Import car sales data, name columns, choose numeric value columns
cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)

continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]


#%% Data cleaning

numeric_cars = numeric_cars.replace('?',np.nan)
numeric_cars = numeric_cars.astype(float)
print(numeric_cars.isnull().sum())

# Because `price` is the column we want to predict, let's remove any rows with missing `price` values
numeric_cars = numeric_cars.dropna(subset=['price'])
print(numeric_cars.isnull().sum())

# Replace missing values in other columns using column means
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
print(numeric_cars.isnull().sum())

# Normalize all columnns to range from 0 to 1 except the target column
price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col


#%% Univariate Model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_name,target_name,df):
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    half = int(len(rand_df)/2)
    train = rand_df.iloc[0:half]
    test = rand_df[half:]
    
    knn = KNeighborsRegressor()
    knn.fit(train[[train_name]], train[target_name])
    predicted_price = knn.predict(test[[train_name]])
    rmse = mean_squared_error(test[target_name], predicted_price)**(1/2)
    return rmse


# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
rmse_results = {}
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val

# Create a Series object from the dictionary so we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


#%% include different k values to minimise rmse
def knn_train_test(train_name,target_name,df):
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    half = int(len(rand_df)/2)
    train = rand_df.iloc[0:half]
    test = rand_df[half:]
    
    k_values = [1, 3, 5, 7, 9]
    k_rmses = {}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train[[train_name]], train[target_name])
        predicted_price = knn.predict(test[[train_name]])
        rmse = mean_squared_error(test[target_name], predicted_price)**(1/2)
        k_rmses[k] = rmse        
    return k_rmses


k_rmse_results = {}
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

# Plot the results
import matplotlib.pyplot as plt

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')


#%% Multivariate Model
    
# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
series_avg_rmse.sort_values()


#Multivariate model continued
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def knn_train_test(train_name,target_name,df):
    np.random.seed(1)
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)
    
    half = int(len(rand_df)/2)
    train = rand_df.iloc[0:half]
    test = rand_df[half:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    k_mapes ={}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train[train_name], train[target_name])
        predicted_price = knn.predict(test[train_name])
        rmse = mean_squared_error(test[target_name], predicted_price)**(1/2)
        mape = mean_absolute_percentage_error(test[target_name], predicted_price)
        k_rmses[k] = rmse    
        k_mapes[k] = mape
    return k_rmses, k_mapes


k_rmse_results = {}
k_mape_results = {}
train_cols_2 = ['horsepower', 'width']
train_cols_3 = ['horsepower', 'width', 'curb-weight']
train_cols_4 = ['horsepower', 'width', 'curb-weight','highway-mpg']
rmse_val, mape_val = knn_train_test(train_cols_2,'price',numeric_cars)
k_rmse_results["2 best features"] = rmse_val
k_mape_results["2 best features"] = mape_val
rmse_val, mape_val = knn_train_test(train_cols_3,'price',numeric_cars)
k_rmse_results["3 best features"] = rmse_val
k_mape_results["3 best features"] = mape_val
rmse_val, mape_val = knn_train_test(train_cols_4,'price',numeric_cars)
k_rmse_results["4 best features"] = rmse_val
k_mape_results["4 best features"] = mape_val
print(k_rmse_results)


for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')


#%%
for k,v in k_mape_results.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('MAPE')



