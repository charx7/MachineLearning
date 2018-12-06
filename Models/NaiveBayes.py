import pandas as pd
import nltk
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from heapq import nlargest

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq, doTf_IDF

def tf_Idf_analysis(df, type):
    print('Im doing tf-idf yeah! f*yeah!!!')
    print('-----Start Tf/TfIdf for the {0} Data ------\n'.format(type))
    # See how many tweets we read
    print("Read {0:d} tweets".format(len(df)))
    raw_tweets = df["text"][:]

    # Do BoW for freq extraction
    ordered_feature_freq_dict, bow_tf, feature_names_tf = doFreq(raw_tweets)

    # Call the tf-idf method
    ordered_idf_dict, bow_tf_idf, feature_names_tf_idf, idf = doTf_IDF(raw_tweets)

    # Get the 10 largest values of the freq dict
    print('\nFor Tf...')
    printHighestFreq(10, ordered_feature_freq_dict)
    # Print the most common tf
    print('\nFor Idf...')
    printHighestFreq(10, ordered_idf_dict)

    print('-----End Tf/Idf for the full Data ------\n')

    return ordered_feature_freq_dict, bow_tf, feature_names_tf, ordered_idf_dict, bow_tf_idf, feature_names_tf_idf, idf

def printHighestFreq(N, ordered_data):
    # Print stuff
    print('The largest ', N ,' word freq for the dataset are: ')
    #for k, v in ordered_feature_freq_dict.items():
    #    print ('%s: %s' % (k, v))
    largest_freq = nlargest(N, ordered_data, key=ordered_data.get)
    # Print the results
    for key in largest_freq:
        print("(",key , ",", ordered_data[key], ")")

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

    # Tf-Idf on the full dataset
    ordered_feature_freq_dict_full, bow_tf_full, feature_names_tf_full, ordered_idf_dict_full, bow_tf_idf_full, feature_names_tf_idf_full, idf_full = tf_Idf_analysis(df, 'full')
    # Tf-Idf on the bot data
    ordered_feature_freq_dict_bot, bow_tf_bot, feature_names_tf_bot, ordered_idf_dict_bot, bow_tf_idf_bot, feature_names_tf_idf_bot, idf_bot = tf_Idf_analysis(botData.head(2000), 'bot')
    # Tf-Idf on the genuine/human data
    ordered_feature_freq_dict_genuine, bow_tf_genuine, feature_names_tf_genuine, ordered_idf_dict_genuine, bow_tf_idf_genuine, feature_names_tf_idf_genuine, idf_genuine = tf_Idf_analysis(genuineData.head(2000), 'genuine')

    # Start the train/test split
    raw_tweets = df['text'][:]

    x = raw_tweets
    y = df['bot'][:]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

    print('The shapes of the train/test split are...\n Train shapes...')
    print(X_train.shape)
    print(y_train.shape)
    print('Test shapes...')
    print(X_test.shape)
    print(y_test.shape)

    # Now prior probability calculation
    # Define our totals for our prior calculations of spam and not spam
    total = 0
    numBot = 0
    # Iteration over the training set
    for tweet in range(X_train.shape[0]):
        # If bot then add one to the botCount
        if y_train.iloc[tweet] == 1:
            numBot += 1
        # Otherwise just add one to the total
        total += 1

    pA = numBot/float(total)
    pNotA = (total - numBot)/float(total)

    print('Bot prior prob is: ', pA)
    print('Not Bot prior prob is: ', pNotA)

    # For posterior, sum tf-idf
    sum_tf_idf_bot = 0
    sum_tf_idf_genuine = 0
    # Sum over all the words for the the TF-IDF
    for word in ordered_feature_freq_dict_genuine:
        tfidf = ordered_feature_freq_dict_genuine[word] * ordered_idf_dict_genuine[word]
        sum_tf_idf_genuine += tfidf

    for word in ordered_feature_freq_dict_bot:
        tfidf = ordered_feature_freq_dict_bot[word] * ordered_idf_dict_bot[word]
        sum_tf_idf_bot += tfidf

    # The test tweet we are gonna use to test our classify XD
    currentTweet = X_train.iloc[5]
    print(currentTweet)
    # declare stemmer
    stemmer = SnowballStemmer("english")

    # Initialize probs
    probNotSpamGivenWords = 1
    probSpamGivenWords = 1
    for word in currentTweet.split():
        # To lower case
        word = word.lower()
        # Stem
        word = stemmer.stem(word)
        print(word)
        # Probability of Spam
        if word in ordered_feature_freq_dict_bot and word in ordered_idf_dict_bot:
            nomin = ordered_feature_freq_dict_bot[word] * ordered_idf_dict_bot[word]
            result = nomin / sum_tf_idf_bot
            print('The probability for the word given bot %s is: %f' %(word, result))
            print(result)
            probSpamGivenWords = probSpamGivenWords * result
        else:
                result = 1

        # Probability of not spam
        if word in ordered_feature_freq_dict_genuine and word in ordered_idf_dict_genuine:
            nomin = ordered_feature_freq_dict_genuine[word] * ordered_idf_dict_genuine[word]
            secondResult = nomin / sum_tf_idf_genuine
            print('The probability for the word given not bot %s is: %f' %(word, result))
            print(secondResult)
            probNotSpamGivenWords = probNotSpamGivenWords * secondResult
        else:
            secondResult = 1

    # Multiply times the prior of being spam pA
    currentTweetBotProb = probSpamGivenWords * pA
    currentTweetNotBotProb = probNotSpamGivenWords * pNotA
    print('The prob of the current tweet being from a bot is: ', currentTweetBotProb)
    print('The prob of the current tweet not being from a bot is: ', currentTweetNotBotProb)

    if (currentTweetBotProb > currentTweetNotBotProb):
        print('The current tweet is from a Bot')
    else:
        print('The current tweet is from a human')
    print('the correct label is: ', y_train.iloc[5])
