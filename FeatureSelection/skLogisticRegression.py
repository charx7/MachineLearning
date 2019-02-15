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

botData = pd.read_csv('../data/datasets_full.csv/traditional_spambots_1.csv/tweets.csv', index_col=0)
genuineData = pd.read_csv('../data/datasets_full.csv/genuine_accounts.csv/tweets.csv', index_col=0)
print('Joining data...')
seed = 42
df = joinData(botData.sample(20000, random_state = seed), genuineData.sample(20000, random_state = seed))

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