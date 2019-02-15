#!/user/bin/env python3
# author : Haibin

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
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#
from Label_Onehot_Encoding_module import LabelEncoder_OneHotEncoder
from Binary_encoding_module import BiEncoder

# Join Data

tqdm.pandas()
print('Loading data...')

# Start Data loading using paralelization parallelLoad function!

#filesRoute = '../data/traditionalSpamBotsChunks1/'
#botData = parallelLoad(filesRoute)
#filesRoute = '../data/genuineTweetsChunks/'
#genuineData = parallelLoad(filesRoute)

botData = pd.read_csv('../data/datasets_full.csv/traditional_spambots_1.csv/tweets.csv', index_col=0)
genuineData = pd.read_csv('../data/datasets_full.csv/genuine_accounts.csv/tweets.csv', index_col=0)

# Joining data

print('Joining data...')

#df = joinData(botData.sample(20000),genuineData.sample(20000))

seed = 42
df = joinData(botData.sample(20000, random_state = seed), genuineData.sample(20000, random_state = seed))

# See how many tweets we read

print("Read {0:d} tweets".format(len(df)))

print('Origin data shape:', df.shape)

print('Origin data columns:\n', list(df.columns))

X_OneHot=LabelEncoder_OneHotEncoder(df)
X_Binary=BiEncoder(df)
y = df['bot']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_OneHot,y,test_size=0.33,random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_Binary,y,test_size=0.33,random_state=42)

# Decission Tree for OneHot encoding

Decission_clf = tree.DecisionTreeClassifier()
Decission_clf.fit(X=X_train1, y=y_train1)


print('\nThe result of OneHot encoding:')
print(f'Score:{Decission_clf.score(X=X_test1,y=y_test1)}')

# Here is an even shorter way of getting the accuracies for each training and test set
# 3-Fold Mean AUC
Cv_scores_3_Fold= cross_val_score(Decission_clf,X_OneHot,y,cv=3)

print(Cv_scores_3_Fold)
# print out the mean cross validation score
print('OneHot encoding 3-Fold Mean :{}'.format(np.mean(Cv_scores_3_Fold)))

# 10-Fold Mean AUC
Cv_scores_10_Fold = cross_val_score(Decission_clf,X_OneHot,y,cv=10)

print(Cv_scores_10_Fold)

print('OneHot encoding 10-Fold Mean :{}'.format(np.mean(Cv_scores_10_Fold)))

# Decission Tree for Binary encoding

Decission_clf2 = tree.DecisionTreeClassifier()
Decission_clf2.fit(X=X_train2, y=y_train2)


print('\nThe result of Binary encoding:')
print(f'Score:{Decission_clf2.score(X=X_test2,y=y_test2)}')

# Here is an even shorter way of getting the accuracies for each training and test set
# 3-Fold Mean AUC
Cv_scores_3_Fold2= cross_val_score(Decission_clf2,X_Binary,y,cv=3)

print(Cv_scores_3_Fold2)
# print out the mean cross validation score
print('Binary encoding 3-Fold Mean :{}'.format(np.mean(Cv_scores_3_Fold2)))

# 10-Fold Mean AUC
Cv_scores_10_Fold2 = cross_val_score(Decission_clf2,X_Binary,y,cv=10)

print(Cv_scores_10_Fold2)

print('Binary encoding 10-Fold Mean :{}'.format(np.mean(Cv_scores_10_Fold2)))


