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