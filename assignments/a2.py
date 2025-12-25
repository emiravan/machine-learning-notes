#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:19:52 2024

@author: root
"""

# Multiple Linear Regression (MLR)
# Polynomial Regression (PR)
# Support Value Regression (SVR)
# Decision Tree (DT)
# Random Forest (RF)

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score

# Import data
datas = pd.read_csv('maaslar_yeni.csv')

# Set datas
# Eğitim
x = datas.iloc[:,2:3] #kolon adı (x2 ve x3 elenmiş hali)
X=x.values #kolon adını sayısal veriye çevirme
# Aranan sonuç (maaş)
y = datas.iloc[:,5:] #kolon adı
Y=y.values #kolon adını sayısal veriye çevirme

# Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# OLS ile eğitim verilerinin hepsine gerçekten ihtiyacımız var mı diye kontrol etme
import statsmodels.api as sm

# Linear regression için OLS
model = sm.OLS(lin_reg.predict(X),X) # X in tahminini X ile karşılaştır
print("-------------------------------------- LINEAR OLS ---------------------------------------\n")
print(model.fit().summary())
# Sonuç = x2 ve x3 ü elersek P value en düşük oluyor
# x2 ve x3 ü eledik
# R2 Degeri: 0.942

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

# Polynomial regression için OLS
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X) # X in tahminini X ile karşılaştır
print("-------------------------------------- POLY OLS ---------------------------------------\n")
print(model2.fit().summary())
# R2 degeri: 0.759

# SVR için verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

# SVR için OLS
print("--------------------------------------- SVR OLS ----------------------------------------\n")
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
# R2 degeri: 0.770

#Decision Tree 
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
    
# DT için OLS
print("--------------------------------------- DT OLS ----------------------------------------\n")
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())
# R2 degeri: 0.751

# Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

# RF için OLS
print("--------------------------------------- RF OLS ----------------------------------------\n")
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
# R2 degeri: 0.719

#Ozet R2 değerleri
print("--------------------------------------- R2 VALUES ----------------------------------------\n")
print('Linear R2')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R2')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print('Decision Tree R2')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2')
print(r2_score(Y, rf_reg.predict(X)))