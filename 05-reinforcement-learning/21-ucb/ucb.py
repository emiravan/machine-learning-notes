#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:54:37 2024

@author: emir
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N= 10000
d=10    
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(10)
    secilenler.append(ad)
    odul=veriler.values[n,ad] #verilerde n. değer = 1 ise ödül 1 , 0 ise ödül 0
    toplam = toplam + odul
    
plt.hist(secilenler)