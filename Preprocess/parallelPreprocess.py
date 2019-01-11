import pandas as pd
import regex as re
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
import multiprocessing as mp
import sys
import time

# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData

# custom analyzer to use in CountVectorizer
class CustomAnalyzer(object):
    def __init__(self):
        self.tknzr_ = TweetTokenizer(strip_handles=True, reduce_len=True)
        self.stemmer_ = SnowballStemmer("english")

    def __call__(self,tweet):
        tokenized_tweet = []
        # clean text from links, references, emojis etc.
        clean_tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet)
        # lowercase for stopwrod removal to work properly
        clean_tweet = clean_tweet.lower()
        # tokenize
        tokenized_tweet = self.tknzr_.tokenize(clean_tweet)
        # stopword removal
        tokenized_tweet = [token for token in tokenized_tweet if token not in stopwords.words('english')]
        # stemming tokens
        tokenized_tweet = [self.stemmer_.stem(token) for token in tokenized_tweet]
        return tokenized_tweet

if __name__ =='__main__':
    start_time = time.time()
    
    # read csv and take only the text
    dfBot = pd.read_csv("../data/traditionalSpamBotsChunks1/tweets_chunk1.csv")
    dfGen = pd.read_csv("../data/genuineTweetsChunks/tweets_chunk1.csv")
    # Join Data
    data = joinData(dfBot, dfGen)
    print("Read {0:d} tweets".format(len(data)))
    raw_tweets = data["text"][:]
    # initialize custom analyzer
    my_analyzer = CustomAnalyzer()

    # parallel HashingVectorizer
    ## using default feature size n_number=2*20
    ## normalization norm='l1'
    ## do not allow for negatives with alternate_sign=False
    ## use custom analyzer
    vctrz = HashingVectorizer(norm='l1', alternate_sign=False ,analyzer=my_analyzer)
    pool = mp.Pool(processes=4) # low CPU utilization... doesn't seem to parallelize shit
    # get a document-term matrix
    bow = pool.map(vctrz.fit_transform, [raw_tweets])
    # get feature namesand calculate word frequencies
    print(bow[0])
    print("--- %s seconds ---" % (time.time() - start_time))
