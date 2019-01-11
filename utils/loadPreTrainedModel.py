import pandas as pd
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.externals import joblib

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq, doTf_IDF, transform_tf

if __name__ == '__main__':
    print('Loading Pre-trained Model...')
    # Load the pre-trained model
    currentModel = joblib.load('../Trained_Models/svm_glove_w2v_model')
    print('Model Loaded Sucess!')
    # Just for testing purposes
    dummy_result = currentModel.predict([[1,1,1,1,1,1,4,1,1,10,1,1,100,1,1,1,1,1,1,1,1,1,1,1,1]])
    print(dummy_result)
