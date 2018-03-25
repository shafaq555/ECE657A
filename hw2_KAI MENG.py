#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:07:26 2018

@author: mac
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing 
import sys
import scipy.spatial.distance as distance


dataset = pd.read_csv('winequality-red.csv', index_col = None, header = 0)
df = dataset.iloc[0:10,:-1]

#1
#minmax normalization
minmax_scaler = preprocessing.MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df))
df_minmax.columns = df.columns

#z-score normalization
std_scaler = preprocessing.StandardScaler()
df_zscore = pd.DataFrame(std_scaler.fit_transform(df))
df_zscore.columns = df.columns

#mean subtract
df_meansubtr = df.copy()
i = 0
while i < np.shape(df)[1]:
    df_meansubtr.iloc[:,i] = df.iloc[:,i] - np.mean(df.iloc[:,i]) 
    i += 1 

#2
#manhatten distance
nearest_dist, farthest_dist = [], []
for i in range(10):
    min_dist, max_dist = sys.maxsize, -1
    for j in range(10):
        if i != j:
            temp = distance.cityblock(df.iloc[i,:], df.iloc[j,:])
            if temp < min_dist:
                min_dist = temp
            if temp > max_dist:
                max_dist = temp
    nearest_dist.append(min_dist)
    farthest_dist.append(max_dist)
nearest_farthest_manhatten = pd.DataFrame(np.column_stack((nearest_dist, farthest_dist)))
nearest_farthest_manhatten.columns = ['nearest', 'farthest']

#euclidean distance
nearest_dist, farthest_dist = [], []
for i in range(10):
    min_dist, max_dist = sys.maxsize, -1
    for j in range(10):
        if i != j:
            temp = distance.euclidean(df.iloc[i,:], df.iloc[j,:])
            if temp < min_dist:
                min_dist = temp
            if temp > max_dist:
                max_dist = temp
    nearest_dist.append(min_dist)
    farthest_dist.append(max_dist)
nearest_farthest_euclidean = pd.DataFrame(np.column_stack((nearest_dist, farthest_dist)))
nearest_farthest_euclidean.columns = ['nearest', 'farthest']

#cosine distance
nearest_dist, farthest_dist = [], []
for i in range(10):
    min_dist, max_dist = sys.maxsize, -1
    for j in range(10):
        if i != j:
            temp = distance.cosine(df.iloc[i,:], df.iloc[j,:])
            if temp < min_dist:
                min_dist = temp
            if temp > max_dist:
                max_dist = temp
    nearest_dist.append(min_dist)
    farthest_dist.append(max_dist)
nearest_farthest_cosine = pd.DataFrame(np.column_stack((nearest_dist, farthest_dist)))
nearest_farthest_cosine.columns = ['nearest', 'farthest']

    

            
