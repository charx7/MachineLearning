import pandas as pd
import nltk
import multiprocessing as mp
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from tqdm import tqdm
from heapq import nlargest

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq, doTf_IDF

def printHighestFreq(N, ordered_data):
    # Print stuff
    print('The largest ', N ,' word freq for the dataset are: ')
    #for k, v in ordered_feature_freq_dict.items():
    #    print ('%s: %s' % (k, v))
    largest_freq = nlargest(N, ordered_data, key=ordered_data.get)
    # Print the results
    print(largest_freq)

if __name__ =='__main__':
    # To measure the progress of our lambda apply functions
    tqdm.pandas()
    print('Loading data...')

    # Start Data loading using paralelization parallelLoad(route_to_files) function!
    filesRoute = '../data/traditionalSpamBotsChunks1/'
    botData = parallelLoad(filesRoute)
    filesRoute = '../data/genuineTweetsChunks/'
    genuineData = parallelLoad(filesRoute)

    print('Joining data...')
    df = joinData(botData.head(1000), genuineData.head(1000))

    print('-----Start Tf/TfIdf for the full Data ------\n')
    # See how many tweets we read
    print("Read {0:d} tweets".format(len(df)))
    raw_tweets = df["text"][:]

    # Do BoW for freq extraction
    ordered_feature_freq_dict_full, bow_full, feature_names_full = doFreq(raw_tweets)

    # Call the tf-idf method
    ordered_tf_idf_dict_full, bow_tf_idf_full, feature_names_tf_idf_full, idf_full = doTf_IDF(raw_tweets)

    # Get the 10 largest values of the freq dict
    print('\nFor Tf...')
    printHighestFreq(10, ordered_feature_freq_dict_full)
    # Print the most common tf
    print('\nFor Tf-Idf...')
    printHighestFreq(10, ordered_tf_idf_dict_full)

    print('-----End Tf/TfIdf for the full Data ------\n')

    print('-----Start Tf/TfIdf for the bot Data ------\n')
    # Compute the bot data frequeies for comparsion
    # See how many tweets does the bot data contain
    print("Read {0:d} tweets of bot data".format(len(botData)))
    raw_bot_tweets = botData.head(1000)["text"][:]

    # Do BoW for freq extraction on bot data
    ordered_feature_freq_dict_bot, bow_bot, feature_names_bot = doFreq(raw_bot_tweets)

    # Call the tf-idf method
    ordered_tf_idf_dict_bot, bow_tf_idf_bot, feature_names_tf_idf_bot, idf_bot = doTf_IDF(raw_bot_tweets)

    # Print the 10 largest ordered data
    print('\nFor Tf...')
    printHighestFreq(10, ordered_feature_freq_dict_bot)
    print('\nFor Tf-Idf...')
    printHighestFreq(10, ordered_tf_idf_dict_bot)
    print('-----End Tf/TfIdf for the bot Data ------\n')
