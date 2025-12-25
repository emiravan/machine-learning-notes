#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:14:00 2024

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
datas = pd.read_csv("maaslar.csv")

#dataframe slice
e=datas.iloc[:,1:2] #egitim seviyesi
m=datas.iloc[:,2:] #maas

#NumPY array conversion
E=e.values
M=m.values

#linear regression
#create linear model
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(E,M)

#polynominal regression for degree 2
#create non-linear model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
e_poly=poly_reg.fit_transform(E)
lin_reg2=LinearRegression()
lin_reg2.fit(e_poly,m)


#polynominal regression for degree 4
poly_reg3 = PolynomialFeatures(degree=4)
e_poly3=poly_reg3.fit_transform(E)
lin_reg3=LinearRegression()
lin_reg3.fit(e_poly3,m)

#visualization
plt.scatter(E,M,color="red")
plt.plot(e,lin_reg.predict(E),color="blue")
plt.show()

plt.scatter(E,M,color="red")
plt.plot(E,lin_reg2.predict(poly_reg.fit_transform(E)),color="blue")
plt.show()

plt.scatter(E,M,color="red")
plt.plot(E,lin_reg3.predict(poly_reg3.fit_transform(E)),color="blue")
plt.show()