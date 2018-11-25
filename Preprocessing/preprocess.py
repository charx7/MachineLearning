import pandas as pd
import regex as re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
# read csv and take only the text
data = pd.read_csv("../data/tweets.csv")
print("Read {0:d} tweets".format(len(data)))
raw_tweets = data["text"][:100]
print(raw_tweets[:10])

tokenized_tweets = []
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
stemmer = PorterStemmer()
# preprocess steps
print("Preprocessing starts \n")
for tweet in raw_tweets:
    # clean text from links, references, emojis etc.
    clean_tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet)
    # tokenize
    tokenized_tweet = tknzr.tokenize(clean_tweet)
    # stopword removal
    for word in tokenized_tweet:
        if word in stopwords.words('english'):
            tokenized_tweet.remove(word)
    # spelling correction and stemming
    for i in range(len(tokenized_tweet)): # stems names!!
        tokenized_tweet[i] = stemmer.stem(tokenized_tweet[i])
    tokenized_tweets.append(tokenized_tweet)    
print("Prerpocessing ends \n")
print("Example result of cleaning and tokenizing: \n")
print(raw_tweets[9])
print(tokenized_tweets[9])



