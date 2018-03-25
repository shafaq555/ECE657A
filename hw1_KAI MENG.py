#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:25:26 2018

@author: mac
"""

# import library
import pandas as pd 
import numpy as np
from scipy.stats import skew
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer 

#1 
wdbc = pd.read_csv('hw1_wdbc.csv', index_col = None, header = None)

wdbc.columns = ['ID', 'Diagnosis', 'radius mean', 'texture mean', 'perimeter mean', 'area mean','smoothness mean', 'compactness mean', 'concavity mean', 'concave points mean', 'symmetry mean', 'fractal dimension mean',
       'radius se', 'texture se', 'perimeter se', 'area se', 'smoothness se', 'compactness se', 'concavity se', 'concave points se', 'symmetry se', 'fractal dimension se',
       'radius worst', 'texture worst',	 'perimeter worst',	'area worst', 'smoothness worst', 'compactness worst', 'concavity worst', 'concave points worst', 'symmetry worst', 'fractal dimension worst']

wdbc_cvf = wdbc.iloc[:, 2:33]

cvf_mean = np.mean(wdbc_cvf)
print(cvf_mean)

cvf_std = np.std(wdbc_cvf)
print(cvf_std)

cvf_var = np.var(wdbc_cvf)
print(cvf_var)

cvf_skew = skew(wdbc_cvf)
print(cvf_skew)

for i in wdbc.columns[2:33]:
    modes = wdbc_cvf[i].mode()
    print(i, modes.values)

#2
for i in wdbc.columns[3:33]:
    r = pearsonr(wdbc['radius mean'], wdbc[i])[0]
    print('PCC of radius mean &', i, ':', r)
    
#3
malignant = []
binign = []
i = 0
while i < len(wdbc['Diagnosis']):
    if wdbc.iloc[i, 1] == 'M':
        malignant.append(wdbc.iloc[i, 2])
        i += 1
    else:
        binign.append(wdbc.iloc[i, 2])
        i += 1

plt.hist(malignant)
plt.title('Histogram of Malignant radius mean')
plt.xlabel('radius mean')
plt.ylabel('Frequency')
plt.show()

plt.hist(binign)
plt.title('Histogram of Binign radius mean')
plt.xlabel('radius mean')
plt.ylabel('Frequency')
plt.show()

#4(For fun)
response = pd.read_csv('ECE 657A W2018 - Class Data (Responses) - Form Responses 1.csv', index_col = None)

# fill in missing values 
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
response.iloc[:, 3:12] = imputer.fit_transform(response.iloc[:, 3:12])

phd = []
i = 0
while i < len(response['Degree Type']):
    if response.iloc[i, 1] == 'PhD':
        j = 1
        phd.append(j)
        i += 1
    else:
        j = 0
        phd.append(j)
        i += 1

r1 = pearsonr(phd, response['SVM'])[0]
    
meng = []
i = 0
while i < len(response['Degree Type']):
    if response.iloc[i, 1] == 'MEng':
        j = 1
        meng.append(j)
        i += 1
    else:
        j = 0
        meng.append(j)
        i += 1

r2 = pearsonr(meng, response['SVM'])[0]
    