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

    print('-----Start Tf/TfIdf for the full Data ------\n')
    # See how many tweets we read
    print("Read {0:d} tweets".format(len(df)))
    raw_tweets = df["text"][:]

    # Do BoW for freq extraction
    ordered_feature_freq_dict_full, bow_full, feature_names_full = doFreq(raw_tweets)

    # Call the tf-idf method
    ordered_idf_dict_full, bow_tf_idf_full, feature_names_tf_idf_full, idf_full = doTf_IDF(raw_tweets)

    # Get the 10 largest values of the freq dict
    print('\nFor Tf...')
    printHighestFreq(10, ordered_feature_freq_dict_full)
    # Print the most common tf
    print('\nFor Idf...')
    printHighestFreq(10, ordered_idf_dict_full)

    print('-----End Tf/Idf for the full Data ------\n')

    print('-----Start Tf/Idf for the bot Data ------\n')
    # Compute the bot data frequeies for comparsion
    # See how many tweets does the bot data contain
    print("Read {0:d} tweets of bot data".format(len(botData)))
    raw_bot_tweets = botData.head(1000)["text"][:]

    # Do BoW for freq extraction on bot data
    ordered_feature_freq_dict_bot, bow_bot, feature_names_bot = doFreq(raw_bot_tweets)

    # Call the tf-idf method
    ordered_idf_dict_bot, bow_tf_idf_bot, feature_names_tf_idf_bot, idf_bot = doTf_IDF(raw_bot_tweets)

    # Print the 10 largest ordered data
    print('\nFor Tf...')
    printHighestFreq(10, ordered_feature_freq_dict_bot)
    print('\nFor Idf...')
    printHighestFreq(10, ordered_idf_dict_bot)
    print('-----End Tf/TfIdf for the bot Data ------\n')

    print('-----Start Tf/Idf for the Genuine Data ------\n')
    # Compute the bot data frequeies for comparsion
    # See how many tweets does the bot data contain
    print("Read {0:d} tweets of genuine data".format(len(genuineData)))
    raw_genuine_tweets = genuineData.head(1000)["text"][:]

    # Do BoW for freq extraction on bot data
    ordered_feature_freq_dict_genuine, bow_genuine, feature_names_genuine = doFreq(raw_genuine_tweets)

    # Call the tf-idf method
    ordered_idf_dict_genuine, bow_tf_idf_genuine, feature_names_tf_idf_genuine, idf_genuine = doTf_IDF(raw_genuine_tweets)

    # Print the 10 largest ordered data
    print('\nFor Tf...')
    printHighestFreq(10, ordered_feature_freq_dict_genuine)
    print('\nFor Idf...')
    printHighestFreq(10, ordered_idf_dict_genuine)
    print('-----End Tf/TfIdf for the Genuine Data ------\n')

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
