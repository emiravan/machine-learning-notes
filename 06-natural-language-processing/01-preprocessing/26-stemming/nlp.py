# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip')
            
import re # Regular expression
import nltk # Natural language tool kit

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

derlem=[]
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum =yorum.lower()
    yorum=yorum.split() 
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)