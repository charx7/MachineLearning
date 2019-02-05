import pandas as pd
from tqdm import tqdm


import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq
from featureEncoding import featureEncoding

tqdm.pandas()
print('Loading data...')

# Start Data loading using paralelization parallelLoad(route_to_files) function!
filesRoute = '../data/traditionalSpamBotsChunks1/'
botData = parallelLoad(filesRoute)
filesRoute = '../data/genuineTweetsChunks/'
genuineData = parallelLoad(filesRoute)

print('Joining data...')
df = joinData(botData.sample(20000), genuineData.sample(20000))

# See how many tweets we read
print("Read {0:d} tweets".format(len(df)))


#######################################SkLearn Logistic Regression####################
cols, X, y = featureEncoding(df)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

clf = LogisticRegression()
clf = clf.fit(X=X_train, y=y_train)

print(f'Selected features are:\n{cols}')
print(f'Accuracy: {clf.score(X=X_test, y=y_test)}') # 1.0