import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData

print('load data...')
botData = pd.read_csv("../data/tweetsBots.csv")
genuineData = pd.read_csv("../data/tweetsGenuine.csv")

df = joinData(botData.head(1000), genuineData.head(1000))

# Drop all columns but the one containing the tweets text
df = df[['text','bot']]

print('do tokenize...\n')

# Do a tokenization by row
df['tokenized_text'] = df.apply(lambda row: word_tokenize(row['text']), axis=1)
# Also the lenght of the tokenizer (could be useful?)
df['tweets_length'] = df.apply(lambda row: len(row['tokenized_text']), axis=1)

# Create an instance of the porter stemmer
stemmer = PorterStemmer()
# Steam the words
df['stemmed_tweets'] = df['tokenized_text'].apply(lambda words:[stemmer.stem(word) for word in words])

# Set-up remove of stop words
stop_words = set(stopwords.words('english'))

# Get the multidimensional array
stemmedWords = df['stemmed_tweets'].values.reshape(-1,).tolist()
# Flatten to 1d array
flattenedStemmedWords = [x for sublist in stemmedWords for x in sublist]

# TODO: Remove the URLS via Regex etc.
print('The flattened stemmed words are: \n', flattenedStemmedWords[:10])
# Cleanup of the stemmed words because they are dirty :O
cleanedStemmedWords = []
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
            "!"
        ] and len(word) > 2 and word not in stop_words:
            cleanedStemmedWords.append(word.lower())
print('The cleaned Stemmed Words are: \n',cleanedStemmedWords[:30])

# Calculate Frequencies
cleanedFrequencies = nltk.FreqDist(cleanedStemmedWords)

for e in cleanedFrequencies.most_common(10):
    print (e)

# TODO XD
print('calculate tf-idf...')
