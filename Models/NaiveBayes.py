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
from preprocess import CustomAnalyzer, doFreq

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
    df = joinData(botData.head(5000), genuineData.head(5000))

    # See how many tweets we read
    print("Read {0:d} tweets".format(len(df)))
    raw_tweets = df["text"][:]

    # Do BoW for freq extraction
    ordered_feature_freq_dict = doFreq(raw_tweets)

    # Print stuff
    print('The largest word freq are: ')
    #for k, v in ordered_feature_freq_dict.items():
    #    print ('%s: %s' % (k, v))

    # Get the 10 largest values of the freq dict
    ten_largest = nlargest(10, ordered_feature_freq_dict,
        key=ordered_feature_freq_dict.get)
    print(ten_largest)
