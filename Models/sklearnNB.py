import pandas as pd
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq, doTf_IDF, transform_tf

if __name__ == '__main__':
    # To difine which method
    USE_MULTINOMIAL = False

    print('Loading data...')
    # Start Data loading using paralelization parallelLoad(route_to_files) function!
    #filesRoute = '../data/traditionalSpamBotsChunks1/'
    #botData = parallelLoad(filesRoute)
    #filesRoute = '../data/genuineTweetsChunks/'
    #genuineData = parallelLoad(filesRoute)

    # Read the english dataz
    botData = pd.read_csv('../data/preprocessedTweets/bot_english_tweets.csv', index_col=0)
    genuineData = pd.read_csv('../data/preprocessedTweets/genuine_english_tweets.csv', index_col=0)

    print('Joining data...')
    seed = 42
    df = joinData(botData.sample(5000, random_state = seed), genuineData.sample(5000, random_state = seed))

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

    # Do BoW for freq extraction
    trainingFull = trainingFull["text"][:]

    # Load common vocab (optional)
    #common_vocab = pd.read_csv('../Preprocess/complete_vocabulary.csv')

    # Rerturn the transformer and vectorizer objects
    X_train_transformed, count_vect, tf_transformer = transform_tf(trainingFull)

    # shape for debugz
    # print('The shape of the transf dataset is: ',X_train_transformed.shape)
    # print('The shape of the training data is: ', trainingFull.shape)
    # print('The shape of the X_train data is: ', X_train.shape)
    # print('The shape of the training labels is: ', y_train.shape)
    # print('The train labels form are: ',y_train.values)

    # Naive Bayes classifier Multinomial or Bernoilli (two classes)
    if USE_MULTINOMIAL == True:
        naive_bayes_classifier = MultinomialNB().fit(X_train_transformed, y_train.values)
    else:
        # With laplace smoothing of alpha = 1 acc goes up to 86% if not then 50%
        naive_bayes_classifier = BernoulliNB(alpha=1.0).fit(X_train_transformed, y_train.values)
    # Score method of the nb classifier
    trainingAcc = naive_bayes_classifier.score(X_train_transformed, y_train.values)
    print('The accuracy of NB on our training data is: ', trainingAcc)

    # convert to array of text values
    test_tweets = X_test.values

    # Perform vector transformation on the test set
    test_tweets_counts = count_vect.transform(test_tweets)
    test_tweets_tfidf = tf_transformer.transform(test_tweets_counts)
    # Run the score function on the transformed test set
    testAcc = naive_bayes_classifier.score(test_tweets_tfidf, y_test.values)
    print('The accuracy of NB on our test set data is: ', testAcc)

    # Test output of the clasifier with fun phrases to be replaced by real tweets
    docs_new = ['Get your free stuff',
                'oh my god taylor mascaras on sale',
                'free indian job get recruited',
                'im slowly dying while doing this project help']
    # Transform the text we are going to test
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tf_transformer.transform(X_new_counts)
    # Call the predict function of the classifier
    predicted = naive_bayes_classifier.predict(X_new_tfidf)

    target_names = ['Human', 'Bot']

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, target_names[category]))

    # Write the model into memory:
    print('Writting Model into memory...')
    with open('../Trained_Models/TfIdf_Nb_Model','wb') as f:
        pickle.dump(naive_bayes_classifier, f)
    print('Sucess model written.')
