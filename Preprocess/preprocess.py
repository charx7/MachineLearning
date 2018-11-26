import pandas as pd
import regex as re
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from typing import List

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
        for word in tokenized_tweet:
            if word in stopwords.words('english'):
                tokenized_tweet.remove(word)
        # stemming tokens
        for i in range(len(tokenized_tweet)): # stems names!!
            tokenized_tweet[i] = self.stemmer_.stem(tokenized_tweet[i])
        return tokenized_tweetnized_tweets

# read csv and take only the text
data = pd.read_csv("tweets.csv")
print("Read {0:d} tweets".format(len(data)))
raw_tweets = data["text"][:100]
# initialize custom analyzer
my_analyzer = CustomAnalyzer()
# Bag of Words
vctrz = CountVectorizer(max_features=1000, vocabulary=None, analyzer=my_analyzer)
bow = vctrz.fit_transform(raw_tweets)
# get feature namesand calculate word frequencies
feature_names = vctrz.get_feature_names()
word_freq = np.true_divide(np.ravel(bow.sum(axis=0)),bow.sum())

feature_freq_dict = {}
testsum = 0
print("Printing tokens with frequency >= {0}".format(0.01))
for i in range(len(feature_names)):
    feature_freq_dict[feature_names[i]] = word_freq[i]
    testsum = testsum + word_freq[i]
    if word_freq[i] >= 0.01:
        print("{0} , {1:3.2f}".format(feature_names[i], word_freq[i]))


