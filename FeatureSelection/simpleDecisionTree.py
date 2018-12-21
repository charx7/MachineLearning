import pandas as pd
from tqdm import tqdm

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq

####################################Join Data##############################################
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


####################################Drop Useless columns##############################################
dataOrigin = df #Origin data
# remove useless features manually
df = df.drop(['bot', 'id','user_id', 'text', 'source', 'in_reply_to_status_id', 
              'in_reply_to_user_id', 'retweeted_status_id'], axis=1)

####################################Split continuous data##############################################
# preprocess feature: favorite_count
df['favorite_count'] = pd.cut(df['favorite_count'], bins=[-1, 20, 40, 60, 80, 100], 
	labels=['favorite_count0_20', 'favorite_count20_40', 'favorite_count40_60', 'favorite_count60_80', 'favorite_count80_100'])
# preprocess feature: retweet_count
df['retweet_count'] = pd.cut(df['retweet_count'], bins=[-1, 20000, 40000, 60000, 300000], 
	labels=['retweet_count0_20k', 'retweet_count20_40k', 'retweet_count40_60k', 'retweet_count60_300k'])


####################################Create dummy variables############################################
cat_vars=['place','num_hashtags','num_urls','num_mentions', 'favorite_count', 'retweet_count']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var,prefix_sep='_',dummy_na=True)
    data1=df.join(cat_list)
    df=data1

cat_vars=['place','num_hashtags','num_urls','num_mentions', 'favorite_count', 'retweet_count']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

####################################Drop nan and strings############################################
cols_to_remove = []

for col in df.columns:
    try:
        _ = df[col].astype(float)
    except ValueError:
        print('Couldn\'t covert %s to float' % col)
        cols_to_remove.append(col)
        pass

# keep only the columns in df that do not contain string
df = df[[col for col in df.columns if col not in cols_to_remove]]

df = df.dropna(axis=1)
df.head()


###########################################Feature Selection########################################
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 10)
rfe = rfe.fit(df, dataOrigin.bot)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


###########################################Selected Features########################################
cols = ['place_nan','num_hashtags_2.0','num_hashtags_3.0','num_hashtags_4.0',
        'num_urls_0.0','num_mentions_1.0','num_mentions_2.0','num_mentions_3.0',
        'favorite_count_favorite_count0_20','retweet_count_retweet_count20_40k']
X = df[cols]
y = dataOrigin['bot']


###########################################Decision Tree########################################
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
clf.fit(X=X_train, y=y_train)
print(f'Feature Importance: {clf.feature_importances_}') # [ 1.,  0.,  0.]
print(f'Score: {clf.score(X=X_test, y=y_test)}') # 1.0


