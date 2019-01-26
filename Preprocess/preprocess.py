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

def embedding_preprocess(corpus):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    stemmer = SnowballStemmer("english")
    preprocessed_corpus = []
    for tweet in corpus:
        # clean
        clean_tweet = cleanString(tweet)
        # tokenize
        tokenized_tweet = tknzr.tokenize(clean_tweet)
        # stopword removal
        tokenized_tweet = [token for token in tokenized_tweet if token not in stopwords.words('english')]
        # remove words with < 3 letters
        tokenized_tweet = [token for token in tokenized_tweet if len(token) >= 3 ]
        # reassemble sentence
        preprosessed_tweet = ' '.join(tokenized_tweet)
        # check if tweet is empty after preprocessing
        if len(preprosessed_tweet) > 0:
            # add preprocessed tweet to preprocessed_corpus
            preprocessed_corpus.append(preprosessed_tweet)
    return preprocessed_corpus

def orderDict(dictToOrder):
    # To order a dictionary
    d_sorted_by_value = OrderedDict(sorted(dictToOrder.items(), key=lambda x: x[1]))
    return d_sorted_by_value

# Function to clean tweet using regex and convert it to lowercase
def cleanString(text):
    # clean text from links, references, emojis etc.
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|((\d+\s)+)|(\d+$)|(RT)|(rt)", "", text)
    # lowercase for stopwrod removal to work properly
    text = text.lower()
    return text

def transform_tf(raw_tweets, **kwargs):
    # Create instance of the analyzer
    my_analyzer = CustomAnalyzer()
    # Get the counts
    if 'vocabulary' in kwargs:
        print('Im using the custom vocab :D')
        count_vect = CountVectorizer(vocabulary=kwargs['vocabulary'], analyzer=my_analyzer)
    else:
        count_vect = CountVectorizer(ngram_range=(1,1), max_features=1500, vocabulary=None, analyzer=my_analyzer)
    # Get them counts
    raw_tweets_counts = count_vect.fit_transform(raw_tweets)
    # Create an object of TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=True).fit(raw_tweets_counts)
    # transform the word counts
    raw_tweets_tf = tf_transformer.transform(raw_tweets_counts)

    return raw_tweets_tf, count_vect, tf_transformer

# Function that does the frequency analysis
def doFreq(raw_tweets, **kwargs):
    my_analyzer = CustomAnalyzer()
    # Bag of Words
    if 'vocabulary' in kwargs:
        vctrz = CountVectorizer(vocabulary=kwargs['vocabulary'], analyzer=my_analyzer)
    else:
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

def doTf_IDF(raw_tweets, **kwargs):
    my_analyzer = CustomAnalyzer()
    # Bag of Words
    if 'vocabulary' in kwargs:
        vctrz = TfidfVectorizer(vocabulary=kwargs['vocabulary'], analyzer=my_analyzer)
    else:
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

# custom analyzer to use in CountVectorizer
class CustomAnalyzer(object):
    def __init__(self):
        self.tknzr_ = TweetTokenizer(strip_handles=True, reduce_len=True)
        self.stemmer_ = SnowballStemmer("english")

    def __call__(self,tweet):
        tokenized_tweet = []
        # clean text from links, references, emojis etc. and convert it to lowercase
        clean_tweet = cleanString(tweet)
        # tokenize
        tokenized_tweet = self.tknzr_.tokenize(clean_tweet)
        # stopword removal
        tokenized_tweet = [token for token in tokenized_tweet if token not in stopwords.words('english')]
        # remove words with < 4 letters
        tokenized_tweet = [token for token in tokenized_tweet if len(token) >= 4 ]
        # stemming tokens
        tokenized_tweet = [self.stemmer_.stem(token) for token in tokenized_tweet]
        return tokenized_tweet

if __name__ == '__main__':
    start_time = time.time()
    # read csv and take only the text
    #dfBot = pd.read_csv("../data/preprocessedTweets/bot_english_tweets.csv")
    #dfGen = pd.read_csv("../data/preprocessedTweets/genuine_english_tweets.csv")
    # Join Data
    #data = joinData(dfBot, dfGen)
    data = pd.read_csv("../data/genuineTweetsChunks/tweets_chunk1.csv")
    print("Read {0:d} tweets".format(len(data)))
    raw_tweets = data["text"].sample(frac=0.1)
    print("Will process {0:d} tweets".format(len(raw_tweets)))
    freq_dict, bow, feature_names = doFreq(raw_tweets)
    i=1
    for key, value in freq_dict.items():
        if i>len(freq_dict)-20:
            print("({0}, {1:10.9f})".format(key,value))
        i = i+1
    print("--- %s seconds ---" % (time.time() - start_time))
