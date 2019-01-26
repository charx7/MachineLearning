import pandas as pd
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.externals import joblib

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq, doTf_IDF, transform_tf
from embedTweet import embedTweet, embed_Dataframe
from embedTweetGlove import loadGloveModel, embed_Dataframe_with_Glove

if __name__ == '__main__':
    # To difine which method
    USE_MULTINOMIAL = False

    print('Loading data...')
    # Read the dataz
    botData = pd.read_csv('../data/preprocessedTweets/bot_english_tweets.csv', index_col=0)
    genuineData = pd.read_csv('../data/preprocessedTweets/genuine_english_tweets.csv', index_col=0)

    print('Joining data...')
    df = joinData(botData.sample(50000), genuineData.sample(50000))

    # Reset indexes after join
    df = df.reset_index()
    # Start the train/test split
    raw_tweets = df['text'][:]

    x = raw_tweets
    y = df['bot'][:]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

    # God Damn Pandas Selection
    trainingIndexes = X_train.index.values
    print('The lenght of the training indexes is: ', len(list(trainingIndexes)))
    indexedDf = df.iloc[trainingIndexes]
    trainingBots = indexedDf.loc[indexedDf['bot'] == 1]
    trainingGenuine = indexedDf.loc[indexedDf['bot'] == 0]
    trainingFull = indexedDf

    start_time = time.time()
    # Load the glove model
    glove_model = loadGloveModel('../Preprocess/glove.twitter.27B.25d.txt')
    print('Finish loading the Glove model!')

    X_train_transformed = embed_Dataframe_with_Glove(X_train, glove_model)
    print('the shape of the X_train_transformed is: ', X_train_transformed.shape)

    # Naive Bayes classifier Multinomial or Bernoilli (two classes)
    if USE_MULTINOMIAL == True:
        naive_bayes_classifier = MultinomialNB().fit(X_train_transformed, y_train.values)
    else:
        # With laplace smoothing of alpha = 1 acc goes up to 86% if not then 50%
        #naive_bayes_classifier = BernoulliNB(alpha=1.0).fit(X_train_transformed, y_train.values)
        naive_bayes_classifier = BernoulliNB(alpha=0.4)
        naive_bayes_classifier.fit(X_train_transformed, y_train.values)

    # Score method of the nb classifier
    trainingAcc = naive_bayes_classifier.score(X_train_transformed, y_train.values)
    print('The accuracy of NB on our training data is: ', trainingAcc)

    # Perform vector transformation on the test set
    test_tweets_w2v = embed_Dataframe_with_Glove(X_test,glove_model)

    # Run the score function on the transformed test set
    testAcc = naive_bayes_classifier.score(test_tweets_w2v, y_test.values)
    print('The accuracy of NB on our test set data is: ', testAcc)
    #
    # Test output of the clasifier with fun phrases to be replaced by real tweets
    docs_new = ['Get your free stuff',
                'oh my god taylor mascaras on sale',
                'free indian job get recruited',
                'im slowly dying while doing this project help']

    # Transform the text we are going to test
    X_new_w2v = embed_Dataframe_with_Glove(docs_new, glove_model)
    # # Call the predict function of the classifier
    predicted = naive_bayes_classifier.predict(X_new_w2v)
    #
    target_names = ['Human', 'Bot']

    for doc, category in zip(docs_new, predicted):
         print('%r => %s' % (doc, target_names[category]))

    # Write the model into memory:
    print('Writting Model into memory...')
    # Model dump
    joblib.dump(naive_bayes_classifier,'../Trained_Models/nb_glove_w2v_model')
    print('Sucess model written.')
