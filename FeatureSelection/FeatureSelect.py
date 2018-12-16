#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:21:40 2018

@author: yifei
"""

import pandas as pd
from tqdm import tqdm

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq

import matplotlib.pyplot as plt 
plt.rc("font", size=14)


############################Join Data############################
tqdm.pandas()
print('Loading data...')

# Start Data loading using paralelization parallelLoad(route_to_files) function!
filesRoute = '../data/traditionalSpamBotsChunks1/'
botData = parallelLoad(filesRoute)
filesRoute = '../data/genuineTweetsChunks/'
genuineData = parallelLoad(filesRoute)

print('Joining data...')
df = joinData(botData.head(20000), genuineData.head(20000))

# See how many tweets we read
print("Read {0:d} tweets".format(len(df)))

print('Origin data shape: ', df.shape)
print('Origin data columns:\n', list(df.columns))

############################Create dummy variables############################
cat_vars=['place','num_hashtags','num_urls','num_mentions']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var,prefix_sep='_',dummy_na=True)
    data1=df.join(cat_list)
    df=data1

cat_vars=['place','num_hashtags','num_urls','num_mentions']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=df[to_keep]
print('After making dummy variables, data columns:\n', data_final.columns.values)



############################Drop useless columns############################
df = data_final
#remove columns with string values
cols_to_remove = []

for col in df.columns:
    try:
        _ = df[col].astype(float)
    except ValueError:
        print('Couldn\'t covert %s to float' % col)
        cols_to_remove.append(col)
        pass

# keep only the columns in df that do not contain string
data = df[[col for col in df.columns if col not in cols_to_remove]]
data = data.drop(['bot','id','user_id'], axis=1)
data = data.dropna(axis=1) #remove columns with nan

############################Feature Selection############################
# Recursive Feature Elimination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(data, df.bot)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)













