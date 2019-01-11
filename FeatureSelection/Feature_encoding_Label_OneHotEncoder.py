#!/user/bin/env python3 i
import pandas as pd 
from tqdm import tqdm
# 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# DictVectorizer
from sklearn.feature_extraction import DictVectorizer
# Cross_val_score
from sklearn.model_selection import cross_val_score
# Imputer 
from sklearn.preprocessing import Imputer 

from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer

import sys
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq
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

# set display columns
# pd.set_option('display.max_columns',1)
# print(pd.options.display.max_columns)
print('Origin data columns:\n', list(df.columns))
# Print df to examinate 
pd.reset_option('display.max_columns')
print('Here is a Pandas DataFrame...')
print('Index is another name for row in Pandas DataFrame. More specific, every row_name should be index_label.')
print(df.head(10))

# Inspect this dataset separately

index = df.index
column = df.columns
values = df.values

print('\n print row_names, it is also index...')
print(index)
print('The exact type of index is',type(index))
print('\n print column name...the beginning Index indicates the type is Index. It is fine.')
print(column)
print('The exact type of column is',type(column))
print('\n Here is value,\n',values)
print(values)
print('\n For value, what is kind of object they are',type(values))

Cleaned_df = df[['id','source','user_id','truncated','in_reply_to_status_id','in_reply_to_user_id','in_reply_to_screen_name','retweeted_status_id','geo','place','contributors','retweet_count','reply_count','favorite_count','favorited','retweeted','possibly_sensitive','num_hashtags','num_urls','num_mentions','created_at','timestamp','crawled_at','updated']]
print(Cleaned_df.head(20))

#Select a column to be a series without using list
Series_for_id = df['id']
print(Series_for_id)

#Compare with using list
DataFrame_for_id_df = df[['id']]
print(DataFrame_for_id_df)
print('Although,this resembles the Series from above, it is technically a DataFrame, a different object!')

# Remove features that just are not useful for us

del Cleaned_df['id']
del Cleaned_df['source']
del Cleaned_df['user_id']
#del Cleaned_df['place']
#del Cleaned_df['contributors']
#del Cleaned_df['retweeted']
#del Cleaned_df['possibly_sensitive']
del Cleaned_df['created_at']
del Cleaned_df['timestamp']
del Cleaned_df['crawled_at']
del Cleaned_df['updated']
print('\n...show head 1000 examples...\n')
print(Cleaned_df.head(1000))

# Exporatory Data Analysis 
# Cleaned_df.info
print('\n\n\n Print intuitive exporatory data analysis...')
print(Cleaned_df.info())
# Replace the empty strings.
#Cleaned_df.replace('','?',inplace = True)
#print('Again Print intuitive exporatory data analysis...')
#print(Cleaned_df.info())

print('It is essential to encoding categorical features into numerical values, right?')

# Missing value table
#missing_values_table(Cleaned_df)

print('Firstly ignore the missing value. Just do next step!')

# Numerical columns

num_cols = ['in_reply_to_status_id','in_reply_to_user_id','in_reply_to_screen_name','retweeted_status_id','retweet_count','reply_count','favorite_count','num_hashtags','num_urls','num_mentions']

# Categorical columns
cate_cols = Cleaned_df.columns.drop(num_cols)
print('\n Print numerical columns...')
print(num_cols)
print('\n Print categorical colums...')
print(cate_cols)
# Convert numerical data
Cleaned_df[num_cols] = Cleaned_df[num_cols].apply(pd.to_numeric,errors='coerce')
print(Cleaned_df.info())
print(Cleaned_df.dtypes)
#numericed_numerical_data = pd.to_numeric(Cleaned_df[num_cols],errors='coerce')
#print(numericed_numerical_data)
# For example df['B'].astype() 
# astype() method is for specific type conversion(i.e. float64,float32,float16)
print('\n After this, Cleaned_df has been converted float and object. it is not only object any more...\n')
print(Cleaned_df.head(1000))
# Check the number of unique values in cate_cols

print('\n')
print(Cleaned_df[cate_cols].nunique(axis=0))
Cleaned_df[cate_cols].apply(lambda x: x.nunique(), axis=0)
print('\n')
print(Cleaned_df[cate_cols].apply(lambda x: x.nunique(),axis=0))
#print(Cleaned_df[cate_cols].apply(lambda x: x.nunique(),axis=1))
print('\n')
print(Cleaned_df[num_cols].apply(lambda x: x.nunique(),axis=0))
print('\n')
#print(Cleaned_df['source'])
#print(Cleaned_df['source'].unique())
#print(Cleaned_df['source'].dtype)

# 
print(Cleaned_df[num_cols])
print(Cleaned_df[cate_cols])
print('\n It is different from original tutorial. Becasuse I do not try to deal with missing value. If it is necessary,do it later...\n')
print(Cleaned_df)
# Impute missing numerical values with using Sklearn.impute SimpleImputer

# define numerical imputer 
# num_imputer = Imputer(strategy='mean')

#num_imputer = SimpleImputer(strategy='median')

# imputing on numerical data
#print(num_imputer.fit_transform(Cleaned_df[num_cols]))
#Cleaned_df[num_cols] = num_imputer.fit_transform(Cleaned_df[num_cols])
#print(Cleaned_df.head(1000))
# define categorical imputer
# cate_imputer = Categorical_Imputer('most_frequent')

# SimpleImputer(strategy='most_frequent')
# imputing on categorical data

# print(cate_imputer.fit_transform(Cleaned_df[cate_cols]))
#Imputed_Cleaned_df_cate_cols=cate_imputer.fit_transform(Cleaned_df[cate_cols])
#print(Imputed_Cleaned_df_cate_cols)

###Get dummies### 

#Dummied_Cleaned_df=pd.get_dummies(Cleaned_df,prefix_sep='_',drop_first=True)

#print(Dummied_Cleaned_df.head(1000))

# Create Boolean Mask
categorical_feature_mask = (Cleaned_df.dtypes==object)
print(categorical_feature_mask)

# filter categorical columns using mask and turn it into a list

categorical_cols=Cleaned_df.columns[categorical_feature_mask].tolist()

print(categorical_cols)

# filter numerical columns using mask and turn it into a list

numerical_cols=Cleaned_df.columns[~categorical_feature_mask].tolist()

print(numerical_cols)
print(Cleaned_df[categorical_cols])
# Numerical Imputer: DataFrameMapper
# Construct numerical imputer
#numeric_imputation_mapper = DataFrameMapper([([numeric_feature],Imputer(strategy="median")) for numeric_feature in numerical_cols],input_df=True,df_out=True)

#Cleaned_df_num = numeric_imputation_mapper.fit_transform(Cleaned_df)

# Categorical Imputer: DataFrameMapper
# Apply categorical imputer
#categorical_imputation_mapper = DataFrameMapper([(category_feature,CategoricalImputer()) for category_feature in categorical_cols],input_df=True,df_out=True)
# imputing categorical missing values
# Cleaned_df_cat=categorical_imputation_mapper.fit_transform(Cleaned_df)

# Build a new dataframe containing only the object columns using .copy() method 
# using copy here so that any changes made in new DataFrame don't get reflected in the original one. 

obj_df=Cleaned_df.select_dtypes(include=['object']).copy()
print(obj_df)
print('The exact type of obj_df_values',type(obj_df.values))
#obj_df[obj_df.isnull().any(axis=1)]
#print(obj_df[obj_df.isnull().any(axis=1)])
print('\n')
obj_df["truncated"].value_counts()
print(obj_df["truncated"].value_counts())
print('\n')
obj_df["geo"].value_counts()
print(obj_df["geo"].value_counts())
print('\n')
obj_df["place"].value_counts()
print(obj_df["place"].value_counts())
print('\n')
obj_df["contributors"].value_counts()
print(obj_df["contributors"].value_counts())
print('\n')
obj_df["favorited"].value_counts()
print(obj_df["favorited"].value_counts())
print('\n')
obj_df["retweeted"].value_counts()
print(obj_df["retweeted"].value_counts())
print('\n')
obj_df["possibly_sensitive"].value_counts()
print(obj_df["possibly_sensitive"].value_counts())

# LabelEncoder 

L_encoder = LabelEncoder()

# Applying LabelEncoder on each of the categorical columns:

obj_df = obj_df.apply(lambda col: L_encoder.fit_transform(col.astype(str)))
print('What is list(L_encoder.classes_)',list(L_encoder.classes_))
print(obj_df)

print('\n')
obj_df["truncated"].value_counts()
print(obj_df["truncated"].value_counts())
print('\n')
obj_df["geo"].value_counts()
print(obj_df["geo"].value_counts())
print('\n')
obj_df["place"].value_counts()
print(obj_df["place"].value_counts())
print('\n')
obj_df["contributors"].value_counts()
print(obj_df["contributors"].value_counts())
print('\n')
obj_df["favorited"].value_counts()
print(obj_df["favorited"].value_counts())
print('\n')
obj_df["retweeted"].value_counts()
print(obj_df["retweeted"].value_counts())
print('\n')
obj_df["possibly_sensitive"].value_counts()
print(obj_df["possibly_sensitive"].value_counts())

# 

print(obj_df.info())
print(obj_df.describe())

print(obj_df['truncated'].values.reshape(-1,1))
print(obj_df['truncated'].values)
print(obj_df.truncated.values)
# OneHotEncoder

#One_encoder = OneHotEncoder(categories='auto',sparse=True)

One_encoder = OneHotEncoder(categories='auto')

#obj_categorical_feature_mask = (obj_df.dtypes==int64) 

#print(obj_categorical_feature_mask)

#One_encoder = OneHotEncoder(obj_categorical_features = obj_categorical_feature_mask,sparse=False)

# fit_transform method expects a 2D array, reshape to transform from 1D to 2D array. 

One_encoded_truncated_array = One_encoder.fit_transform(obj_df.truncated.values.reshape(-1,1)).toarray()

# One_encoded_truncated_feature = One_encoder.fit_transform(obj_df[['truncated']].reshape(-1,1)).toarray() DataFrame object has no attribute 'reshape'

print('Print One_encoded_truncated_array is:\n',One_encoded_truncated_array)
print('The type of One_encoded_truncated_array:',type(One_encoded_truncated_array))
print(One_encoded_truncated_array.shape)
print(One_encoded_truncated_array.shape[1])
for i in range(One_encoded_truncated_array.shape[1]):
    print('The value of i is:',i)

dfOnehot = pd.DataFrame(One_encoded_truncated_array,columns=["Truncated_"+ str(int(i)) for i in range(One_encoded_truncated_array.shape[1])])

One_hot_encoded_df = pd.concat([obj_df,dfOnehot],axis=1)

print(One_hot_encoded_df)
#truncated_feature_labels = list(L_encoder.classes_)

#truncated_feature = pd.DataFrame(truncated_feature_arr, columns=truncated_feature_labels)
#print(truncated_feature)
print(len(obj_df.columns))
for j in range(len(obj_df.columns)):
    location_indicator=j
    print(location_indicator)
    indicator_for_cat=obj_df.columns[location_indicator]
    print(indicator_for_cat)
print(obj_df.columns[0])

#def One_encoder_function(input_obj_df):
   # for j in range(len(input_obj_df.columns)):
    #    location_indicator=j
     #   indicator_for_cat=input_obj_df.columns[location_indicator]
      #  One_encoded_cat_col_array = One_encoder.fit_transform(input_obj_df[str(indicator_for_cat)].values.reshape(-1,1)).toarray()
       # for k in range(One_encoded_cat_col_array.shape[1]):
        #    label_for_indicator_for_cat=str(indicator_for_cat)+'_'+str(int(k))
         #   print('Current columns label is',label_for_indicator_for_cat)
          #  print('Current array for current label is\n',One_encoded_cat_col_array)
           # print('\n')
            #Current_cat_features_labels=label_for_indicator_for_cat for k in range(One_encoded_cat_col_array.shape[1])]
            # print(Current_cat_features_labels)
            #current_cat_dataframe=pd.DataFrame(One_encoded_cat_col_array,columns=[str(indicator_for_cat)+'_'+str(int(k))])
        
       # Current_cat_dataframe=pd.DataFrame(One_encoded_cat_col_array,columns=[str(indicator_for_cat)+'_'+str(int(m)) for m in range(One_encoded_cat_col_array.shape[1])])
       # print(Current_cat_dataframe) 
        
        
    
    #Current_cat_total_df=pd.DataFrame(One_encoded_cat_col_array,columns=Current_cat_features_labels)
    #return Current_cat_total_df 

   # return Current_cat_dataframe
  #  return (0)

def One_encoder_function(input_obj_df_column):
    
    indicator_for_cat = input_obj_df_column.name
    
    One_encoded_cat_array = One_encoder.fit_transform(input_obj_df_column.values.reshape(-1,1)).toarray()
    
    Current_cat_dataframe = pd.DataFrame(One_encoded_cat_array,columns=[str(indicator_for_cat)+'_'+str(int(n)) for n in range(One_encoded_cat_array.shape[1])])

    return Current_cat_dataframe

truncated_One_encoded_dataframe = One_encoder_function(obj_df['truncated'])

#print(truncated_One_encoded_dataframe)

geo_One_encoded_dataframe = One_encoder_function(obj_df['geo'])

#print(geo_One_encoded_dataframe)

place_One_encoded_dataframe = One_encoder_function(obj_df['place'])

#print(place_One_encoded_dataframe)

contributors_One_encoded_dataframe = One_encoder_function(obj_df['contributors'])

#print(contributors_One_encoded_dataframe)

favorited_One_encoded_dataframe = One_encoder_function(obj_df['favorited'])

#print(favorited_One_encoded_dataframe)

retweeted_One_encoded_dataframe = One_encoder_function(obj_df['retweeted'])

#print(retweeted_One_encoded_dataframe)

possibly_sensitive_One_encoded_dataframe = One_encoder_function(obj_df['possibly_sensitive'])

#print(possibly_sensitive_One_encoded_dataframe)

Current_cats_dataframe = pd.concat([truncated_One_encoded_dataframe,geo_One_encoded_dataframe,place_One_encoded_dataframe,contributors_One_encoded_dataframe,favorited_One_encoded_dataframe,retweeted_One_encoded_dataframe,possibly_sensitive_One_encoded_dataframe],axis=1)

#print(Current_cats_dataframe)

total_One_encoded_obj_df = pd.concat([obj_df,Current_cats_dataframe],axis=1)

print(total_One_encoded_obj_df)

#for j in range(len(obj_df.columns)):
    
 #   location_indicator=j
 #   cat_name_index=obj_df.columns[j]
 #   current_cat_dataframe = One_encoder_function(obj_df[str(cat_name_index)])
 #   print(current_cat_dataframe)
#total_cat_dataFrame=One_encoder_function(obj_df)

#print(total_cat_dataFrame)

#Ohe_truncated_feature = One_encoder_function(obj_df[['truncated']])
    
#print(Ohe_truncated_feature)

#Apply OneHotEncoder on categorical feature columns
#One_hot_df = One_encoder.fit_transform(obj_df)

#obj_df = obj_df.apply(lambda labelEncoded_col : One_encoder.fit_transform(obj_df))
#print(One_hot_df)

# Concatenate Numerical Feature and Categorical Feature

#Encoded_Cleaned_df_Concatenated = pd.concat([Cleaned_df[numerical_cols],total_One_encoded_obj_df],axis=1)
Encoded_Cleaned_df_Concatenated = pd.concat([Cleaned_df[numerical_cols],Current_cats_dataframe],axis=1).drop(['in_reply_to_screen_name'],axis=1)


print(Encoded_Cleaned_df_Concatenated)
print(np.isnan(Encoded_Cleaned_df_Concatenated))
print(np.where(np.isnan(Encoded_Cleaned_df_Concatenated)))
print(np.where(np.isnan(Cleaned_df[num_cols])))
print(np.isnan(Cleaned_df[num_cols]))

X = Encoded_Cleaned_df_Concatenated

y = df['bot']

print(df['bot'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

Decission_clf = tree.DecisionTreeClassifier()
Decission_clf.fit(X=X_train, y=y_train)

#print(f'Feature Importance:{Decission_clf.feature_importance_}')

print(f'Score:{Decission_clf.score(X=X_test,y=y_test)}')

Cv_scores = cross_val_score(Decission_clf,X,y,scoring='roc_auc', cv=3)
print(Cv_scores)
# print out the mean cross validation score

print('3-Fold Mean AUC:{}'.format(np.mean(Cv_scores)))

# 10-Fold Mean AUC

Cv_scores_10_Fold = cross_val_score(Decission_clf,X,y,scoring='roc_auc',cv=10)

print(Cv_scores_10_Fold)

print('10-Fold Mean AUC:{}'.format(np.mean(Cv_scores_10_Fold)))
