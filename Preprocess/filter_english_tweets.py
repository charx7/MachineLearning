import pandas as pd
import time
import regex as re
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool

def detect_language(text, wordlist):
    # check NaN cases
    if pd.isnull(text):
        return False
    stmr = SnowballStemmer("english")
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    try:
        clean_text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|((\d+\s)+)|(\d+$)|(RT)|(rt)", "", text)
    except:
        print("===========================================")
        print("ERROR ON REGEX WHILE PROCESSING TWEET TEXT: \n")
        print(text)
        print("===========================================")
    tokenized_text = tknzr.tokenize(clean_text.lower())
    t = [token for token in tokenized_text if stmr.stem(token) in wordlist]
    if len(tokenized_text)>0:
        return (len(t)/len(tokenized_text) > 0.5)
    else:
        return False

def filter_tweets(tweets):
    stmr = SnowballStemmer("english")
    wordlist = [stmr.stem(word) for word in words.words()]
    english_tweets = []
    for tweet in tweets:
        if detect_language(tweet, wordlist):
            english_tweets.append(tweet);
    return english_tweets

if __name__ == '__main__':
    start_time = time.time()
    # read chosen csv
    data = pd.read_csv("../data/genuineTweetsChunks/tweets.csv")
    print("Read {0:d} tweets".format(len(data)))
    # define dataset
    raw_tweets = data["text"].sample(frac=0.05)
    print("Will process {0:d} tweets".format(len(raw_tweets)))
    # split dataset to parts to paralellize language detection
    n = 100
    split_tweets = [raw_tweets[i:i + n] for i in range(0, len(raw_tweets), n)]
    # filter out non-english tweets
    pool = Pool(processes=8)
    english_tweets = pool.map(filter_tweets, split_tweets)
    # flatten resulting list of lists to one list
    english_tweets = [item for sublist in english_tweets for item in sublist]
    print("English tweets found after filtering: {0:d}".format(len(english_tweets)))
    print("Exporting results to csv file")
    df = pd.DataFrame(english_tweets, columns=['text'])
    df.to_csv('genuine_english_tweets.csv')
    print("--- %s seconds ---" % (time.time() - start_time))
    
