# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv',header=None)

satir_sayisi = veriler.shape[0] # Satır sayısı hesaplama

t=[]
for i in range(0,satir_sayisi):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori
kurallar =apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)

print(list(kurallar))