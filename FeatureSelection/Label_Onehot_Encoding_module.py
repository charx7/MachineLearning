#Implement for LabelEncoder and OneHotEncoder

# author :Haibin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def LabelEncoder_OneHotEncoder(Whole_Dataframe):

    df =Whole_Dataframe
    Cleaned_df = df[['id', 'source', 'user_id', 'truncated', 'in_reply_to_status_id', 'in_reply_to_user_id',
                     'in_reply_to_screen_name', 'retweeted_status_id', 'geo', 'place', 'contributors', 'retweet_count',
                     'reply_count', 'favorite_count', 'favorited', 'retweeted', 'possibly_sensitive', 'num_hashtags',
                     'num_urls', 'num_mentions', 'created_at', 'timestamp', 'crawled_at', 'updated']]
    Whole_Dataframe = df
    del Cleaned_df['id']
    del Cleaned_df['source']
    del Cleaned_df['user_id']
    del Cleaned_df['created_at']
    del Cleaned_df['timestamp']
    del Cleaned_df['crawled_at']
    del Cleaned_df['updated']

    num_cols = ['in_reply_to_status_id', 'in_reply_to_user_id', 'in_reply_to_screen_name', 'retweeted_status_id',
                'retweet_count', 'reply_count', 'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions']

    # Convert numerical data
    Cleaned_df[num_cols] = Cleaned_df[num_cols].apply(pd.to_numeric, errors='coerce')

    # Select categorical columns using object feature
    obj_df = Cleaned_df.select_dtypes(include=['object']).copy()

    # LabelEncoder
    L_encoder = LabelEncoder()

    # Applying LabelEncoder on each of the categorical columns:
    obj_df = obj_df.apply(lambda col: L_encoder.fit_transform(col.astype(str)))
    print(obj_df)

    # OneHotEncoder
    One_encoder = OneHotEncoder(categories='auto')

    def One_encoder_function(input_obj_df_column):
        indicator_for_cat = input_obj_df_column.name

        One_encoded_cat_array = One_encoder.fit_transform(input_obj_df_column.values.reshape(-1, 1)).toarray()

        Current_cat_dataframe = pd.DataFrame(One_encoded_cat_array,
                                             columns=[str(indicator_for_cat) + '_' + str(int(n)) for n in
                                                      range(One_encoded_cat_array.shape[1])])

        return Current_cat_dataframe

    truncated_One_encoded_dataframe = One_encoder_function(obj_df['truncated'])

    geo_One_encoded_dataframe = One_encoder_function(obj_df['geo'])

    place_One_encoded_dataframe = One_encoder_function(obj_df['place'])

    contributors_One_encoded_dataframe = One_encoder_function(obj_df['contributors'])

    favorited_One_encoded_dataframe = One_encoder_function(obj_df['favorited'])

    retweeted_One_encoded_dataframe = One_encoder_function(obj_df['retweeted'])

    possibly_sensitive_One_encoded_dataframe = One_encoder_function(obj_df['possibly_sensitive'])

    Current_cats_dataframe = pd.concat(
        [truncated_One_encoded_dataframe, geo_One_encoded_dataframe, place_One_encoded_dataframe,
         contributors_One_encoded_dataframe, favorited_One_encoded_dataframe, retweeted_One_encoded_dataframe,
         possibly_sensitive_One_encoded_dataframe], axis=1)
    print(Current_cats_dataframe)

    Encoded_Cleaned_df_Concatenated = pd.concat([Cleaned_df[num_cols], Current_cats_dataframe], axis=1).drop(
        ['in_reply_to_screen_name'], axis=1)

    LabelandOneHot_Encoded_feature = Encoded_Cleaned_df_Concatenated

    return LabelandOneHot_Encoded_feature