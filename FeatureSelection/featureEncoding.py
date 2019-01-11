#############################
## Feature Encoding
#############################
import pandas as pd

def featureEncoding(df):
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

###########################################Selected Features########################################
	cols = ['place_nan','num_hashtags_2.0','num_hashtags_3.0','num_hashtags_4.0',
	        'num_urls_0.0','num_mentions_1.0','num_mentions_2.0','num_mentions_3.0',
	        'favorite_count_favorite_count0_20','retweet_count_retweet_count20_40k']
	X = df[cols]
	y = dataOrigin['bot']

	return (cols, X, y)