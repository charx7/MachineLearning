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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt
from sklearn.preprocessing import StandardScaler
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

LR_clf = LogisticRegression(penalty='l2',C=1.0,class_weight=None,solver='warn')

LR_clf.fit(X_train,y_train)

y_pred = LR_clf.predict(X_test)

acc=mt.accuracy_score(y_test,y_pred)

print("Accuracy",acc)

# Here is an even shorter way of getting the accuracies for each training and test set

Cv_scores_accuracy=cross_val_score(LR_clf,X,y,cv=3)
print(Cv_scores_accuracy)

weights = LR_clf.coef_.T # take transpose to make a column vector

variable_names = X.columns

for coef, name in sorted(zip(weights,variable_names)):

    print(name,'has weight of', coef[0])

print('\n')

# Processing with Normalization

# normalize the features based on the mean and standard deviation of each column.

Scaler = StandardScaler()

Scaler.fit(X_train) # find scaling for each column that make this zero mean and unit std
                    # the line of code only looks at training data to get mean and std and we can use it to transform new feature data.
Scaled_X_train = Scaler.transform(X_train)

Scaled_X_test = Scaler.transform(X_test) #apply those means and std to the test set.

LR_clf_for_ScaledData = LogisticRegression(penalty='l2',C=1.0,class_weight=None,solver='lbfgs')

LR_clf_for_ScaledData.fit(Scaled_X_train,y_train)

y_pred_1 = LR_clf_for_ScaledData.predict(Scaled_X_test)

acc_1 = mt.accuracy_score(y_test,y_pred_1)

print('Accuracy_1',acc_1)

weights_1 = LR_clf_for_ScaledData.coef_.T # take transpose to make a column vector

variable_names = X.columns

for coef, name in sorted(zip(weights_1,variable_names)):

    print(name,'has weight of', coef[0])

from matplotlib import pyplot as plt

plt.style.use('ggplot')

Drawing_Weights = pd.Series(LR_clf_for_ScaledData.coef_[0],index=X.columns)
Drawing_Weights.plot(kind='bar')
plt.show()

