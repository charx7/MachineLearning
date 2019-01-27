import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import preprocessing
from tqdm import tqdm
import time
import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from preprocess import CustomAnalyzer, doFreq, doTf_IDF, transform_tf
from embedTweet import embed_Dataframe
from embedTweetGlove import loadGloveModel, embed_Dataframe_with_Glove

if __name__ == '__main__':
    print('Hi ill be a support vector machine :D! Using GLOVE(STANDFORD) embeds :O')

    # Which kernel to use for the svm?
    USE_LINEAR_KERNEL = True

    # Read the dataz
    botData = pd.read_csv('../data/preprocessedTweets/bot_english_tweets.csv', index_col=0)
    genuineData = pd.read_csv('../data/preprocessedTweets/genuine_english_tweets.csv', index_col=0)

    print('Joining data...')
    df = joinData(botData.sample(2000), genuineData.sample(2000))

    # Reset indexes after join
    df = df.reset_index()
    # Start the train/test split
    raw_tweets = df['text'][:]

    x = raw_tweets
    y = df['bot'][:]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)
    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=43)

    # God Damn Pandas Selection
    trainingIndexes = X_train.index.values
    indexedDf = df.iloc[trainingIndexes]
    trainingBots = indexedDf.loc[indexedDf['bot'] == 1]
    trainingGenuine = indexedDf.loc[indexedDf['bot'] == 0]
    trainingFull = indexedDf

    # Do BoW for freq extraction
    trainingFull = trainingFull["text"][:]
    # Rerturn the transformer and vectorizer objects
    print('-----Init tf transform on the dataset to a vector form------')
    print('-----Using w2v----------------------------------------------\n')

    # Time it
    start_time = time.time()
    # Load the glove model
    glove_model = loadGloveModel('../Preprocess/glove.twitter.27B.100d.txt')
    print('Finish loading the Glove model!')

    # Begin the Glove embed
    X_train_transformed = embed_Dataframe_with_Glove(X_train, glove_model)
    print('The shape of the transformation is: ',X_train_transformed.shape)

    print('-----End tf transform on the dataset to a vector form----\n')

    print('---Start transform on the validation set tweets----------\n')
    # Perform vector transformation on the validation set
    val_tweets_w2v = embed_Dataframe_with_Glove(X_val, glove_model)
    print('---End transform on the validation set tweets------------\n')

    print('---Start transform on the test set tweets----------\n')
    # Perform vector transformation on the validation set
    test_tweets_w2v = embed_Dataframe_with_Glove(X_test, glove_model)
    print('---End transform on the test set tweets------------\n')
    print("--- %s seconds ---" % (time.time() - start_time))

    # hyper parameter values for the svm radial kernel
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

    # Grid Search (BruteForce) over the hyper params
    best_score = 0
    # Declare a dictionary that will store the best values
    best_params = {'C': None, 'gamma': None, 'bestModel': None}
    print('------Init hyperparameter Optimization-----')
    # Timer
    start_time = time.time()
    if USE_LINEAR_KERNEL == True:
        print('Using Linear Kernel...\n Scaling data...')
        # Scale the data is neccesary for a linear kernel
        # Create an scaler object
        training_scaler = preprocessing.MinMaxScaler().fit(X_train_transformed)
        # Fit the scaler object that just used train data to everything else
        X_train_transformed = training_scaler.transform(X_train_transformed)
        val_tweets_w2v = training_scaler.transform(val_tweets_w2v)
        test_tweets_w2v = training_scaler.transform(test_tweets_w2v)
        print('Init hyperparam optimization...')
        for C in tqdm(C_values):
            svc = svm.LinearSVC(C=C, max_iter = 70000)
            svc.fit(X_train_transformed, y_train.values)
            score = svc.score(val_tweets_w2v, y_val.values)

            if score > best_score:
                best_score = score
                best_params['C'] = C
                best_params['bestModel'] = svc
    else:
        print('Using the rbf kernel...')
        for C in tqdm(C_values):
            for gamma in gamma_values:
                svc = svm.SVC(C=C, gamma=gamma)
                svc.fit(X_train_transformed, y_train.values)
                score = svc.score(val_tweets_w2v, y_val.values)

                if score > best_score:
                    best_score = score
                    best_params['C'] = C
                    best_params['gamma'] = gamma
                    best_params['bestModel'] = svc

    # Score the final model on the test set separated at the beginning
    # Retreive the best model from the best_params
    test_score = best_params['bestModel'].score(test_tweets_w2v, y_test.values)

    print("--- %s seconds ---" % (time.time() - start_time))
    print('\nThe best hyperparameters on the grid are: ', best_params)
    print('\nScoring: ', best_score, ' on the validation set.')
    print('\nScoring: ', test_score, ' on the test set.')

    # To dump the model into the trained model folder
    print('\nSaving the best model...')
    joblib.dump(best_params['bestModel'],'../Trained_Models/svm_glove_w2v_model')
