#Implement for BiEncoder

# author : Haibin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import category_encoders as ce


def BiEncoder(Whole_Dataframe):

    df =Whole_Dataframe
    Cleaned_df = df[['source', 'user_id', 'truncated','in_reply_to_status_id', 'in_reply_to_user_id',
                     'retweeted_status_id', 'geo','place', 'contributors', 'retweet_count',
                     'reply_count', 'favorite_count', 'favorited', 'retweeted', 'possibly_sensitive', 'num_hashtags',
                     'num_urls', 'num_mentions', 'created_at', 'timestamp', 'crawled_at', 'updated']]

    del Cleaned_df['source']
    del Cleaned_df['user_id']
    del Cleaned_df['created_at']
    del Cleaned_df['timestamp']
    del Cleaned_df['crawled_at']
    del Cleaned_df['updated']

    num_cols = ['in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id',
                'retweet_count', 'reply_count', 'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions']

    cate_cols = Cleaned_df.columns.drop(num_cols)

    Cleaned_df[cate_cols] = Cleaned_df[cate_cols].astype('object')


    # Convert numerical data

    Cleaned_df[num_cols] = Cleaned_df[num_cols].apply(pd.to_numeric, errors='coerce')

    # Select categorical columns using object feature

    obj_df = Cleaned_df.select_dtypes(include=['object']).copy()


    #Applying BinaryEncoder on the whole dataframe

    encoder=ce.BinaryEncoder()
    obj_df_changed_bina=encoder.fit_transform(obj_df)

    print('\n The index for Nan value before imputing: \n')
    print(np.where(np.isnan(Cleaned_df[num_cols])))

    #define numerical imputer

    num_imputer=SimpleImputer(missing_values=np.nan, strategy='median')
    # Imputing on numerical data
    Cleaned_df[num_cols]=num_imputer.fit_transform(Cleaned_df[num_cols])

    print('\n The index for Nan value after imputing: \n')
    print(np.where(np.isnan(Cleaned_df[num_cols])))

    # Combine numerical features and Binary encoded features

    Biencoded_feature=pd.concat([Cleaned_df[num_cols], obj_df_changed_bina], axis=1)

    return(Biencoded_feature)





