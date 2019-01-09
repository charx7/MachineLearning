import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
import time
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# custom imports
from preprocess import cleanString

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
        # remove words with < 4 letters
        tokenized_tweet = [token for token in tokenized_tweet if len(token) >= 4 ]
        # reassemble sentence
        preprosessed_tweet = ' '.join(tokenized_tweet)
        # check if tweet is empty after preprocessing
        if len(preprosessed_tweet) > 0:
            # add preprocessed tweet to preprocessed_corpus
            preprocessed_corpus.append(preprosessed_tweet)
    return preprocessed_corpus
        
# Stop words simple function for the demo
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    return results

# function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

if __name__ == '__main__':
    # read data
    data = pd.read_csv("../data/traditionalSpamBotsChunks1/tweets_chunk1.csv")
    print("Read {0:d} tweets".format(len(data)))
    raw_tweets = data["text"].head(100)
    print("Will process {0:d} tweets".format(len(raw_tweets)))
    clean_corpus = preprocess_corpus(raw_tweets);
    
    # Corpus for the demo
    corpus = ['king is a strong man',
              'queen is a wise woman',
              'boy is a young man',
              'girl is a young woman',
              'prince is a young king',
              'princess is a young queen',
              'man is strong',
              'woman is pretty',
              'prince is a boy will be king',
              'princess is a girl will be queen']
    # Get the clean Corpus
    corpus = remove_stop_words(corpus)
    # define maximum length of sentence
    max_sentence_length = max([len(x.split(" ")) for x in clean_corpus])
    #make vocab processor
    vocab_processor = tflearn.data_utils.VocabularyProcessor(max_sentence_length)
    # our sentences represented as indices instead of words
    integerized_sentences = list(vocab_processor.fit_transform(clean_corpus))
    
    #set our vocab size
    vocab_size = len(vocab_processor.vocabulary_)
    # get word-to-integer dictionary from vocabulary processor
    vocab_dict = vocab_processor.vocabulary_._mapping
    # store word-to-integer dictionary as a Tensor to retrieve when importing model
    tf_vocabulary = tf.Variable(list(vocab_dict.keys()), name = 'vocabulary')
    tf_integerized_vocabulary = tf.Variable(list(vocab_dict.values()), name = 'integerized_vocabulary')
    # form skipgrams
    WINDOW_SIZE = 2
    data = []
    for sentence in integerized_sentences:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1]:
                if neighbor != word:
                    data.append([word, neighbor])
    #for text in clean_corpus:
    #    print(text)
    df = pd.DataFrame(data, columns = ['input', 'label'])
    
    # vocabulary size to use in setting up the parameters for the training
    ONE_HOT_DIM = len(vocab_processor.vocabulary_)
    print("Vocabulary size: {}".format(ONE_HOT_DIM))
    # store vocabulary size as a Tensor to retrieve when importing model
    tf_vocabulary_size = tf.Variable(ONE_HOT_DIM, name = 'vocabulary_size')
    
    X = [] # input word
    Y = [] # target word

    for x, y in zip(df['input'], df['label']):
        X.append(to_one_hot_encoding(x))
        Y.append(to_one_hot_encoding(y))

    # convert them to numpy arrays
    X_train = np.asarray(X)
    Y_train = np.asarray(Y)

    # making placeholders for X_train and Y_train
    x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM), name = 'one_hot_input')
    y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

    # word embedding will be 2 dimension for 2d visualization
    EMBEDDING_DIM = 50

    # hidden layer: which represents word vector eventually
    W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]), name = 'W1')
    b1 = tf.Variable(tf.random_normal([1]), name = 'b1') #bias
    hidden_layer = tf.add(tf.matmul(x,W1), b1, name = 'w2v')

    # output layer
    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
    b2 = tf.Variable(tf.random_normal([1]))
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

    # loss function: cross entropy
    loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

    # training operation
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Add ops to save and restore variables
    saver = tf.train.Saver()
    start_time = time.time()
    # Start the training
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    iteration = 20000
    for i in range(iteration):
        # input is X_train which is one hot encoded word
        # label is Y_train which is one hot encoded neighbor word
        sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
        if i % 3000 == 0:
            print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))
    print("--- %s seconds ---" % (time.time() - start_time))
    save_path = saver.save(sess, "tf_saved_models\\word_emb")
    print("Model saved in path: %s" % save_path)
    # Now the hidden layer (W1 + b1) is actually the word look up table
    vectors = sess.run(W1 + b1)
    print(vectors)

    #cols = ['x{}'.format(i) for i in range(1,EMBEDDING_DIM+1)]
    #w2v_df = pd.DataFrame(vectors, columns = cols)
    #w2v_df['word'] = vocab_dict.keys()
    #w2v_df = w2v_df[['word', cols]]
    #print('The output vector model is: \n',w2v_df)
