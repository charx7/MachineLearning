#!/user/bin/env python3
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#

import sys
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq
from Label_Onehot_Encoding_module import LabelEncoder_OneHotEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Join Data

tqdm.pandas()
print('Loading data...')

# Start Data loading using paralelization parallelLoad function!

filesRoute = '../data/traditionalSpamBotsChunks1/'
botData = parallelLoad(filesRoute)
filesRoute = '../data/genuineTweetsChunks/'
genuineData = parallelLoad(filesRoute)

# Joining data

print('Joining data...')

df = joinData(botData.head(5000),genuineData.head(5000))

# See how many tweets we read

print("Read {0:d} tweets".format(len(df)))

print('Origin data shape:', df.shape)

print('Origin data columns:\n', list(df.columns))

X=LabelEncoder_OneHotEncoder(df)

y = df['bot']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

# Decission Tree

Decission_clf = tree.DecisionTreeClassifier()
Decission_clf.fit(X=X_train, y=y_train)

print(f'Score:{Decission_clf.score(X=X_test,y=y_test)}')

# Here is an even shorter way of getting the accuracies for each training and test set
# 3-Fold Mean AUC
Cv_scores_3_Fold= cross_val_score(Decission_clf,X,y,cv=3)

print(Cv_scores_3_Fold)
# print out the mean cross validation score
print('3-Fold Mean AUC:{}'.format(np.mean(Cv_scores_3_Fold)))
#####################################################################################################
# 10-Fold Mean AUC

Cv_scores_10_Fold = cross_val_score(Decission_clf,X,y,cv=10)

print(Cv_scores_10_Fold)

print('10-Fold Mean AUC:{}'.format(np.mean(Cv_scores_10_Fold)))
