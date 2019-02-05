import pandas as pd
from tqdm import tqdm
from featureEncoding import featureEncoding

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq

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



###########################################SkLearn Decision Tree########################################
cols, X, y = featureEncoding(df)

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

clf = tree.DecisionTreeClassifier()


clf = clf.fit(X=X_train, y=y_train)

print(f'Selected features are:\n{cols}')
print(f'Features inportance:\n{clf.feature_importances_}') # [ 1.,  0.,  0.]
print(f'Accuracy: {clf.score(X=X_test, y=y_test)}') # 1.0


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=cols, 
                                class_names='bot', filled=True, rounded=True, 
                               special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("skTwitterBot", view=True)