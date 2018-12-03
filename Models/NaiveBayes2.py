import pandas as pd
import nltk
import multiprocessing as mp
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from tqdm import tqdm

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad

def tokenize(df):
    # To measure the progress of our lambda apply functions
    # Need to specify since they will be running on separate processes
    tqdm.pandas()
    # Do a tokenization by row
    #print('Tokenizing text...')
    #print(df.loc[69377:69380,['text']]) # This will have NA as text
    # Drop the NA tweets texts so we dont have problems with our tokenizers
    df = df.dropna(subset=['text'])
    # Do the apply method
    #df['tokenized_text'] = df.progress_apply(lambda row: word_tokenize(row['text']), axis=1)
    df['tokenized_text'] = df.apply(lambda row: word_tokenize(row['text']), axis=1)
    # Return df
    return df

def wordCount(df):
    # Also the lenght of the tokenizer (could be useful?)
    #print('Getting number of words...')
    #df['tweets_length'] = df.progress_apply(lambda row: len(row['tokenized_text']), axis=1)
    df['tweets_length'] = df.apply(lambda row: len(row['tokenized_text']), axis=1)
    # Return the new df
    return df

def steem(df):
    #print('Stemming Words...')
    # Create an instance of the porter stemmer
    stemmer = PorterStemmer()
    # Steam the words
    #df['stemmed_tweets'] = df['tokenized_text'].progress_apply(lambda words:[stemmer.stem(word) for word in words])
    df['stemmed_tweets'] = df['tokenized_text'].apply(lambda words:[stemmer.stem(word) for word in words])
    # Return the new stemmed df
    return df

def removeUrls(df):
    #print('Removing Urls...')
    # Remove the urls/# etc
    #df['stemmed_tweets'] = df['stemmed_tweets'].progress_apply(lambda words:[ re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", word) for word in words])
    df['stemmed_tweets'] = df['stemmed_tweets'].apply(lambda words:[ re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", word) for word in words])
    # Return the df without URLs
    return df

def removeStopWords(df):
    # Set-up remove of stop words
    stop_words = set(stopwords.words('english'))

    # Get the multidimensional array
    stemmedWords = df['stemmed_tweets'].values.reshape(-1,).tolist()
    # Flatten to 1d array
    #print('Flattening the array...')
    flattenedStemmedWords = [x for sublist in stemmedWords for x in sublist]

    #print('The flattened stemmed words are: \n', flattenedStemmedWords[:10])
    # Cleanup of the stemmed words because they are dirty :O
    cleanedStemmedWords = []
    #print('Removing stop words and punctuation...')
    for word in flattenedStemmedWords:
        # Not commas periods and applause.
        if word not in [
                ",",
                ".",
                "``",
                "''",
                ";",
                "?",
                "--",
                ")",
                "(",
                ":",
                "!",
                "...",
                "http",
                "u2013"
            ] and len(word) > 2 and word not in stop_words:
                cleanedStemmedWords.append(word.lower())
    #print('The cleaned Stemmed Words are: \n',cleanedStemmedWords[:30])
    return cleanedStemmedWords

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
    df = joinData(botData.head(50000), genuineData.head(5000))

    # Drop all columns but the one containing the tweets text
    df = df[['text','bot']]

    # Divide data into chunks
    n = 1000  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

    # Use 4 processes
    pool = mp.Pool(8) # use 4 processes

    print('Tokenizing text...')
    # Create a list of async functions
    funclist = []
    for df in list_df:
        # Process each df using and async function
        f = pool.apply_async(tokenize, [df])
        # Append it to a list of async functions
        funclist.append(f)

    result = []
    for f in tqdm(funclist):
        # Timeout in 2 mins
        # Use the get method on the f object generated by apply_async
        # to retrive the result once the process is finished
        result.append(f.get(timeout=120))

    # Concat results
    df = pd.concat(result)
    # Divide data into chunks for parallel processing
    n = 1000  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

    print('Counting number of words...')
    # Create a list of async functions
    funclist = []
    for df in list_df:
        # Process each df using and async function
        f = pool.apply_async(wordCount, [df])
        # Append it to a list of async functions
        funclist.append(f)

    result = []
    for f in tqdm(funclist):
        # Timeout in 2 mins
        # Use the get method on the f object generated by apply_async
        # to retrive the result once the process is finished
        result.append(f.get(timeout=120))

    # Concat results
    df = pd.concat(result)

    print('Stemming...')
    # Divide data into chunks for parallel processing
    n = 1000  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

    # Create a list of async functions
    funclist = []
    for df in list_df:
        # Process each df using and async function
        f = pool.apply_async(steem, [df])
        # Append it to a list of async functions
        funclist.append(f)

    result = []
    for f in tqdm(funclist):
        # Timeout in 2 mins
        # Use the get method on the f object generated by apply_async
        # to retrive the result once the process is finished
        result.append(f.get(timeout=120))

    # Concat results
    df = pd.concat(result)

    print('Removing Urls...')
    # Divide data into chunks for parallel processing
    n = 1000  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

    # Create a list of async functions
    funclist = []
    for df in list_df:
        # Process each df using and async function
        f = pool.apply_async(removeUrls, [df])
        # Append it to a list of async functions
        funclist.append(f)

    result = []
    for f in tqdm(funclist):
        # Timeout in 2 mins
        # Use the get method on the f object generated by apply_async
        # to retrive the result once the process is finished
        result.append(f.get(timeout=120))

    # Concat results
    df = pd.concat(result)

    print('Removing Stop Words...')
    # Divide data into chunks for parallel processing
    n = 1000  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

    # Create a list of async functions
    funclist = []
    for df in list_df:
        # Process each df using and async function
        f = pool.apply_async(removeStopWords, [df])
        # Append it to a list of async functions
        funclist.append(f)

    result = []
    for f in tqdm(funclist):
        # Timeout in 2 mins
        # Use the get method on the f object generated by apply_async
        # to retrive the result once the process is finished
        result.append(f.get(timeout=120))

    print('The final results are: \n')
    print(result[0][1:10])
    # # Do the tokenization step
    # df = tokenize(df)
    #
    # # Do the word count
    # df = wordCount(df)
    #
    # # Do the stemmer
    # df = steem(df)
    #
    # # Removal of URLs
    # df = removeUrls(df)
    #
    # # Remove Stop words
    # cleanedStemmedWords = removeStopWords(df)
    #
    # # Calculate Frequencies
    # cleanedFrequencies = nltk.FreqDist(cleanedStemmedWords)
    #
    # print('The cleaned freq are: \n',cleanedFrequencies)
    #
    # for e in cleanedFrequencies.most_common(10):
    #     print (e)
    #
    # # TODO XD
    # print('calculate tf-idf...')
    print('...End')
