import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
import time
import sys
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# custom imports
from preprocess import cleanString
# User defined Imports ugly python import syntax >:(
sys.path.append('../Preprocess')
from dataJoin import joinData

# preprocess corpus
def preprocess_corpus(corpus):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    stemmer = SnowballStemmer("english")
    preprocessed_corpus = []
    for tweet in corpus:
        # clean
        clean_tweet = cleanString(tweet)
        # tokenize
        tokenized_tweet = tknzr.tokenize(clean_tweet)
        # stopword removal
        tokenized_tweet = [token for token in tokenized_tweet if token not in stopwords.words('english')]
        # remove words with < 3 letters
        tokenized_tweet = [token for token in tokenized_tweet if len(token) >= 3 ]
        # reassemble sentence
        preprosessed_tweet = ' '.join(tokenized_tweet)
        # check if tweet is empty after preprocessing
        if len(preprosessed_tweet) > 0:
            # add preprocessed tweet to preprocessed_corpus
            preprocessed_corpus.append(preprosessed_tweet)
        # Add case for empty tweets after the transformation TODO(revision needed)
        else:
            # Append the 'EMPTY AFTER PREPROCESS' token
            preprocessed_corpus.append('EMPTY_AFTER_PREPROCESS')

    return preprocessed_corpus

# define w2v model
def build_word2vec():
    # store information in Tensors to retrieve when importing model
    tf_vocabulary = tf.Variable(list(vocab_dict.keys()), name = 'vocabulary')
    tf_integerized_vocabulary = tf.Variable(list(vocab_dict.values()), name = 'integerized_vocabulary')
    tf_embedding_size = tf.Variable(embedding_size, name = 'embedding_size')

    #Pivot Words
    x = tf.placeholder(tf.int32, shape=[None,], name="x_pivot_idxs")
    #Target Words
    y = tf.placeholder(tf.int32, shape=[None,], name="y_target_idxs")

    ## Make our word embedding matrix
    Embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                            name="word_embedding")


    #Weights and biases for NCE Loss
    nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                  stddev=tf.sqrt(1/embedding_size)),
                            name="nce_weights")
    nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")


    #Look up pivot word embedding
    pivot = tf.nn.embedding_lookup(Embedding, x, name="word_embed_lookup")


    #expand the dimension and set shape
    train_labels = tf.reshape(y, [tf.shape(y)[0], 1])

    ##Compute Loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                         biases  = nce_biases,
                                         labels  = train_labels,
                                         inputs  = pivot,
                                         num_sampled = num_samples,
                                         num_classes = vocab_size,
                                         num_true = 1))

    ##Create optimizer
    optimizer = tf.contrib.layers.optimize_loss(loss,
                                                tf.train.get_global_step(),
                                                learning_rate,
                                                "Adam",
                                                clip_gradients=5.0,
                                                name="Optimizer")
    sesh = tf.Session()

    sesh.run(tf.global_variables_initializer())

    return optimizer, loss, x, y, sesh

if __name__ == '__main__':
    # Read the dataz
    botData = pd.read_csv('../data/preprocessedTweets/bot_english_tweets.csv', index_col=0)
    genuineData = pd.read_csv('../data/preprocessedTweets/genuine_english_tweets.csv', index_col=0)
    print('Joining data...')
    data = joinData(botData.sample(10000, random_state=42), genuineData.sample(10000, random_state=42))
    # How many tweets are in the full dataset
    print("Read {0:d} tweets".format(len(data)))
    # Clear memory for eficiency
    del botData
    del genuineData
    # How many tweets are we taking for the embeddings
    raw_tweets = data["text"].sample(20000)
    print("Will process {0:d} tweets".format(len(raw_tweets)))

    # Corpus to use
    clean_corpus = preprocess_corpus(raw_tweets);
    # Print shape for debug
    #print('The shape of the corpus is: \n',np.array(clean_corpus).shape)

    # define maximum length of sentence
    max_sentence_length = max([len(x.split(" ")) for x in clean_corpus])
    #make vocab processor
    vocab_processor = tflearn.data_utils.VocabularyProcessor(max_sentence_length)
    # our sentences represented as indices instead of words
    integerized_sentences = list(vocab_processor.fit_transform(clean_corpus))
    print(">> Vocabulary size: {}".format(len(vocab_processor.vocabulary_)))

    #set our vocab size
    vocab_size = len(vocab_processor.vocabulary_)
    # get word-to-integer dictionary from vocabulary processor
    vocab_dict = vocab_processor.vocabulary_._mapping

    # form skipgrams
    WINDOW_SIZE = 2
    data = []
    for sentence in integerized_sentences:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1]:
                if neighbor != word:
                    data.append([word, neighbor])
    df = pd.DataFrame(data, columns = ['input', 'label'])

    # form training set
    X_train = [] # input word
    y_train = [] # target word
    for x, y in zip(df['input'], df['label']):
        X_train.append(x)
        y_train.append(y)


    # Size of our embedding matrix
    embedding_size = 25
    # Number of samples for NCE Loss
    num_samples = 64
    # Learning Rate
    learning_rate = 0.001
    # initialize model
    optimizer, loss, x, y, sesh = build_word2vec()

    batch_size = 100
    num_epochs = 15
    # Num batches in training set
    num_batches = len(X_train) // batch_size
    print(">> Number of batches: {}".format(num_batches))
    # create saver to save our weights
    saver = tf.train.Saver()

    start_time = time.time()

    for e in range(num_epochs):
        print(">> EPOCH: {}".format(e+1))
        for i in range(num_batches):
            if i != range(num_batches-1):
                x_batch = X_train[i*batch_size:i * batch_size + batch_size]
                y_batch = y_train[i*batch_size:i * batch_size + batch_size]
            else:
                x_batch = X_train[i*batch_size:]
                y_batch = y_train[i*batch_size:]

            _, l = sesh.run([optimizer, loss], feed_dict = {x: x_batch, y: y_batch})

            if (i>0 and i %100 == 0) or i==num_batches-1:
                print("BATCH", i, "of", num_batches-1, "LOSS:", l)
        save_path = saver.save(sesh, "tf_saved_models\\word_emb")

    print("--- %s seconds ---" % (time.time() - start_time))
