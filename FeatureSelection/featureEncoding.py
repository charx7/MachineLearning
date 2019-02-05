#############################
## Feature Encoding
#############################
import pandas as pd

def featureEncoding(df):
	#df = dataOrigin#############################################################################################
	dataOrigin = df #Origin data
	# remove useless features manually
	df = df.drop(['bot', 'id','user_id', 'text', 'source','reply_count'], axis=1)

	#truncated
	df.truncated.loc[~df.truncated.isnull()] = 'Notnan'  # not nan
	#df.truncated.loc[df.truncated.isnull()] = 0   # nan
	#possibly_sensitive
	df.possibly_sensitive.loc[~df.possibly_sensitive.isnull()] = 'Notnan'  # not nan
	#df.possibly_sensitive.loc[df.possibly_sensitive.isnull()] = 0   # nan

	####################################Create dummy variables############################################
	cat_vars=['place','truncated','possibly_sensitive','geo','contributors','favorited','retweeted']
	for var in cat_vars:
	    cat_list='var'+'_'+var
	    cat_list = pd.get_dummies(df[var], prefix=var,prefix_sep='_',dummy_na=True)
	    data1=df.join(cat_list)
	    df=data1

	cat_vars=['place','truncated','possibly_sensitive','geo','contributors','favorited','retweeted']
	data_vars=df.columns.values.tolist()
	to_keep=[i for i in data_vars if i not in cat_vars]

	df = df[to_keep]
	########


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

	###########################################Feature Selection########################################
	# Recursive Feature Elimination
	from sklearn.feature_selection import RFE
	from sklearn.linear_model import LogisticRegression

	# create a base classifier used to evaluate a subset of attributes
	model = LogisticRegression()
	# create the RFE model and select 3 attributes
	rfe = RFE(model, 15)
	rfe = rfe.fit(df, dataOrigin.bot)
	# summarize the selection of the attributes
	print(rfe.support_)
	print(rfe.ranking_)
	sup = rfe.support_

	col_temp = []
	index = 0
	for c in sup:
	    if c == True:
	        col_temp.append(df.columns[index])
	    index += 1

	###########################################Selected Features########################################
	cols = col_temp
	X = df[cols]
	y = dataOrigin['bot']

	return (cols, X, y)