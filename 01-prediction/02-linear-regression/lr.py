# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1. Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Data Preprocessing
#2.1. Load Data
veriler = pd.read_csv('satislar.csv')
print(veriler)

aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

satislar2 = veriler.iloc[:, :1].values
print(satislar2)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

# Optional: Scale data
'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)
'''

# Build and train the model (Linear Regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions
tahmin = lr.predict(x_test)

# Sort the index of training data for plotting
x_train = x_train.sort_index()
y_train = y_train.sort_index()

# Plot the training data
plt.plot(x_train, y_train)

plt.title("aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")