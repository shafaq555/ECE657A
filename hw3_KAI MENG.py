#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:07:26 2018

@author: Kai Meng
"""

# import library
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# load dataset 
dataset = pd.read_csv('crime dataset.csv', index_col = None, header = None)
df = dataset.iloc[:,:-1]

# deal with missing values '?'
df = df.replace('?', 'NaN')
df = df.drop(df.columns[3], axis = 1)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
df = imputer.fit_transform(df)

# feature scaling
sc = StandardScaler()
df = sc.fit_transform(df)

# compute covariance matrix
df = pd.DataFrame(df)
covma = df.cov()

# get eigen values and eigen vectors, as well as the required output
eigen_values, eigen_vectors = np.linalg.eig(covma)
eigen_values_desc = sorted(eigen_values, reverse = True)
eigen_values_20 = eigen_values_desc[0:20]
output = pd.DataFrame(eigen_values_20)
output.to_csv('output.csv')

# determine the cut off point
summ = sum(eigen_values)

pc = []
a = 0
for item in eigen_values:
    pc.append(item)
    a = a + item
    if a/summ >= 0.95:
        print(pc)
        print('number of principal components:', len(pc))
        print('variance explained:', sum(pc)/summ)
        break
    else:
        continue

