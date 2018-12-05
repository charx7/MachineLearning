import pandas as pd
import regex as re
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from tqdm import tqdm
import sys
from collections import OrderedDict
import time

# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData

def orderDict(dictToOrder):
    # To order a dictionary
    d_sorted_by_value = OrderedDict(sorted(dictToOrder.items(), key=lambda x: x[1]))
    return d_sorted_by_value

# Function that does the frequency analysis
def doFreq(raw_tweets):
    my_analyzer = CustomAnalyzer()
    # Bag of Words
    vctrz = CountVectorizer(max_features=1000, vocabulary=None, analyzer=my_analyzer)
    bow = vctrz.fit_transform(raw_tweets)
    # get feature names and calculate word frequencies
    feature_names = vctrz.get_feature_names()
    word_freq = np.true_divide(np.ravel(bow.sum(axis=0)),bow.sum())

    # Construct a dictionary that will be the return of the function
    feature_freq_dict = {}
    for i in tqdm(range(len(feature_names))):
        # Inserta a k,v pair with key = word and value the freq
        feature_freq_dict[feature_names[i]] = word_freq[i]

    # Call the order dict method
    ordered_feature_freq_dict = orderDict(feature_freq_dict)

    # Print Vocab size
    print('vocabulary size of tf: {}'.format(len(vctrz.vocabulary_)))
    # Return the orderred dict and the BoW
    return ordered_feature_freq_dict, bow, feature_names

def doTf_IDF(raw_tweets):
    my_analyzer = CustomAnalyzer()
    # Bag of Words
    vctrz = TfidfVectorizer(max_features=1000, vocabulary=None, analyzer=my_analyzer)
    bow = vctrz.fit_transform(raw_tweets)
    # get feature names and calculate word frequencies
    feature_names = vctrz.get_feature_names()
    # Get the IDF
    idf = vctrz.idf_
    word_freq = np.true_divide(np.ravel(bow.sum(axis=0)),bow.sum())

    tf_idf_dict = dict(zip(vctrz.get_feature_names(), idf))

    # Call the order dict method
    ordered_tf_idf_dict = orderDict(tf_idf_dict)

    # Print Vocab size
    print('vocabulary size of tf-idf: {}'.format(len(vctrz.vocabulary_)))
    # Return the orderred dict and the BoW
    return ordered_tf_idf_dict, bow, feature_names, idf

    # Returns
    return ordered_tf_idf_dict, bow, feature_names, idf

# custom analyzer to use in CountVectorizer
class CustomAnalyzer(object):
    def __init__(self):
        self.tknzr_ = TweetTokenizer(strip_handles=True, reduce_len=True)
        self.stemmer_ = SnowballStemmer("english")

    def __call__(self,tweet):
        tokenized_tweet = []
        # clean text from links, references, emojis etc.
        clean_tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|((\d+\s)+)|(\d+$)|(RT)|(rt)", "", tweet)
        # lowercase for stopwrod removal to work properly
        clean_tweet = clean_tweet.lower()
        # tokenize
        tokenized_tweet = self.tknzr_.tokenize(clean_tweet)
        # stopword removal
        tokenized_tweet = [token for token in tokenized_tweet if token not in stopwords.words('english')]
        # stemming tokens
        tokenized_tweet = [self.stemmer_.stem(token) for token in tokenized_tweet]
        return tokenized_tweet

if __name__ == '__main__':
    start_time = time.time()
    # read csv and take only the text
    #dfBot = pd.read_csv("../data/tweetsBots.csv")
    #dfGen = pd.read_csv("../data/tweetsGenuine.csv")
    # Join Data
    #data = joinData(dfBot, dfGen)
    data = pd.read_csv("../data/traditionalSpamBotsChunks1/tweets.csv")
    print("Read {0:d} tweets".format(len(data)))
    raw_tweets = data["text"].sample(frac=0.2)
    print("Will process {0:d} tweets".format(len(raw_tweets)))
    freq_dict = doFreq(raw_tweets)
    i=1
    for key, value in freq_dict.items():
        if i>len(freq_dict)-20:
            print("({0}, {1:10.9f})".format(key,value))
        i = i+1
    print("--- %s seconds ---" % (time.time() - start_time))
