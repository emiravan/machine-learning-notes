# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip')
yorumlar = yorumlar.dropna() # NaN değerleri içeren satırları kaldırın

import re # Regular expression
import nltk # Natural language tool kit

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing
derlem = []
for i in range(len(yorumlar)):
    try:
        yorum = re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][i])
        yorum = yorum.lower()
        yorum = yorum.split()
        yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
        yorum = ' '.join(yorum)
        derlem.append(yorum)
    except Exception as e:
        print(f"Error {i} line: {e}")


# Feature Extraction
# Bag of words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()  # Bağımsız değişken
y = yorumlar.iloc[:,1].values  # Bağımlı değişken

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)   

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train) # X_train'den y_traini öğren demek (eğitim)

y_pred = gnb.predict(X_test) # Tahmin çıkar

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
