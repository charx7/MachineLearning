import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from tqdm import tqdm

import sys
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData

print('Loading data...')
botData = pd.read_csv("../data/tweetsBots.csv")
genuineData = pd.read_csv("../data/tweetsGenuine.csv")

print('Joining data...')
df = joinData(botData, genuineData)

# Drop all columns but the one containing the tweets text
df = df[['text','bot']]

print('Do tokenize...\n')

# To measure the progress of our lambda apply functions
tqdm.pandas()

# Do a tokenization by row
print('Tokenizing text...')
#print(df.loc[69377:69380,['text']]) # This will have NA as text
# Drop the NA tweets texts so we dont have problems with our tokenizers
df = df.dropna(subset=['text'])

df['tokenized_text'] = df.progress_apply(lambda row: word_tokenize(row['text']), axis=1)

# Also the lenght of the tokenizer (could be useful?)
print('Getting number of words...')
df['tweets_length'] = df.progress_apply(lambda row: len(row['tokenized_text']), axis=1)

print('Stemming Words...')
# Create an instance of the porter stemmer
stemmer = PorterStemmer()
# Steam the words
df['stemmed_tweets'] = df['tokenized_text'].progress_apply(lambda words:[stemmer.stem(word) for word in words])

# Remove the urls/# etc
df['stemmed_tweets'] = df['stemmed_tweets'].progress_apply(lambda words:[ re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", word) for word in words])

# Set-up remove of stop words
stop_words = set(stopwords.words('english'))

# Get the multidimensional array
stemmedWords = df['stemmed_tweets'].values.reshape(-1,).tolist()
# Flatten to 1d array
print('Flattening the array...')
flattenedStemmedWords = [x for sublist in tqdm(stemmedWords) for x in sublist]

# TODO: Remove the URLS via Regex etc.
print('The flattened stemmed words are: \n', flattenedStemmedWords[:10])
# Cleanup of the stemmed words because they are dirty :O
cleanedStemmedWords = []
print('Removing stop words and punctuation...')
for word in tqdm(flattenedStemmedWords):
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
print('The cleaned Stemmed Words are: \n',cleanedStemmedWords[:30])

# Calculate Frequencies
cleanedFrequencies = nltk.FreqDist(cleanedStemmedWords)

print('The cleaned freq are: \n',cleanedFrequencies)

for e in cleanedFrequencies.most_common(10):
    print (e)

# TODO XD
print('calculate tf-idf...')
