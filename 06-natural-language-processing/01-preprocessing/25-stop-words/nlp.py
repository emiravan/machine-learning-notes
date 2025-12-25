# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip')
            
import re # Regular expression

yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][0])
yorum =yorum.lower()
yorum=yorum.split() 

import nltk # Natural language tool kit

durma = nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()