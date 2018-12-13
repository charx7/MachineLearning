import pandas as pd
import regex as re
import numpy as np
import sys
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from collections import OrderedDict
from nltk.corpus import words
from preprocess import CustomAnalyzer, doFreq, doTf_IDF
from dataJoin import joinData

# read bot data
bot_data = pd.read_csv("../data/traditionalSpamBotsChunks1/bot_english_tweets.csv")
print("Read {0:d} bot tweets".format(len(bot_data)))
raw_bot_tweets = bot_data["text"]
print("Will process {0:d} bot tweets".format(len(raw_bot_tweets)))

# read genuine data
genuine_data = pd.read_csv("../data/genuineTweetsChunks/genuine_english_tweets.csv")
print("Read {0:d} genuine tweets".format(len(genuine_data)))
raw_genuine_tweets = genuine_data["text"]
print("Will process {0:d} genuine tweets".format(len(raw_genuine_tweets)))

# getting bot vocabulary
start_time = time.time()
bot_freq_dict, bot_bow, bot_feature_names, bot_idf = doTf_IDF(raw_bot_tweets)
print("--- %s seconds ---" % (time.time() - start_time))
print("BOT term frequencies (top 20)")
i=1
for key, value in bot_freq_dict.items():
    if i<20:
        print("({0}, {1:10.9f})".format(key,value))
    i = i+1

# getting genuine vocabulary
start_time = time.time()
genuine_freq_dict, genuine_bow, genuine_feature_names, genuine_idf = doTf_IDF(raw_genuine_tweets)
print("--- %s seconds ---" % (time.time() - start_time))
print("GENUINE term frequencies (top 20)")
i=1
for key, value in genuine_freq_dict.items():
    if i<20:
        print("({0}, {1:10.9f})".format(key,value))
    i = i+1

# combine vocabularis
complete_vocab = bot_feature_names + list(set(genuine_feature_names) - set(bot_feature_names))
print("Common vocabulary of length {0} \n".format(len(complete_vocab)))
# export to csv
try:
    df = pd.DataFrame(complete_vocab, columns=['words'])
    df.to_csv('complete_vocabulary.csv')
except:
    print("FAILED TO EXPORT VOCABULARY TO CSV")

# read data for bot and genuine and test common vocabulary run
dfBot = pd.read_csv("../data/genuineTweetsChunks/tweets_chunk1.csv")
dfGen = pd.read_csv("../data/traditionalSpamBotsChunks1/tweets_chunk1.csv")
# Join Data
join_data = joinData(dfBot, dfGen)
test_tweets = join_data["text"]
print("Will test on {0} mixed tweets".format(len(test_tweets)))
start_time = time.time()
freq_dict, bow, feature_names, idf = doTf_IDF(test_tweets, vocabulary=complete_vocab)
print("--- %s seconds ---" % (time.time() - start_time))
print("TEST term frequencies (top 20)")
i=1
for key, value in freq_dict.items():
    if i<20:
        print("({0}, {1:10.9f})".format(key,value))
    i = i+1
