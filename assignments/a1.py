#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:48:40 2024

@author: root
"""

#libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data processing
        
datas = pd.read_csv("odev_tenis.csv")
print(datas)

#extract columns

temperature=datas[["temperature"]]
print(temperature)

humidity = datas[["humidity"]]
print(humidity)

# Encoding categorical data to numeric

from sklearn import preprocessing
    
datas2=datas.apply(preprocessing.LabelEncoder().fit_transform)

temperature = datas2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
temperature=ohe.fit_transform(temperature).toarray()
print(temperature)

weather = pd.DataFrame(data=temperature, index=range(14), columns=['o', 'r', 's'])
lastDatas=pd.concat([weather,datas.iloc[:,1:3]],axis=1)
lastDatas=pd.concat([datas2.iloc[:,-2:],lastDatas],axis=1)

#training datas and split for testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(lastDatas.iloc[:,:-1],lastDatas.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#backward elimination

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=lastDatas.iloc[:,:-1],axis=1)

X_l=lastDatas.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model =sm.OLS(lastDatas.iloc[:,-1:],X_l).fit()
print(model.summary())

# Eliminate 0th column and do the same thing

lastDatas=lastDatas.iloc[:,1:]

X = np.append(arr=np.ones((14,1)).astype(int),values=lastDatas.iloc[:,:-1],axis=1)

X_l=lastDatas.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
model =sm.OLS(lastDatas.iloc[:,-1:],X_l).fit()
print(model.summary())

#backward elimination ends.change model (eliminate 0th columns)

x_test=x_test.iloc[:,1:]
x_train=x_test.iloc[:,1:]

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)