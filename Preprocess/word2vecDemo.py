import pandas as pd
import numpy as np
import tensorflow as tf

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
    # Define an empty set of words
    words = []
    # Construct the word set
    for text in corpus:
        for word in text.split(' '):
            words.append(word)
    # Word set that will have the word vector output
    words = set(words)
    # Generate a label for each word using skip gram
    word2int = {}

    for i,word in enumerate(words):
        word2int[word] = i

    sentences = []
    for sentence in corpus:
        sentences.append(sentence.split())

    WINDOW_SIZE = 2

    data = []
    for sentence in sentences:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1]:
                if neighbor != word:
                    data.append([word, neighbor])
    for text in corpus:
        print(text)
    df = pd.DataFrame(data, columns = ['input', 'label'])

    ONE_HOT_DIM = len(words)

    X = [] # input word
    Y = [] # target word

    for x, y in zip(df['input'], df['label']):
        X.append(to_one_hot_encoding(word2int[ x ]))
        Y.append(to_one_hot_encoding(word2int[ y ]))

    # convert them to numpy arrays
    X_train = np.asarray(X)
    Y_train = np.asarray(Y)

    # making placeholders for X_train and Y_train
    x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM), name = 'one_hot_input')
    y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

    # word embedding will be 2 dimension for 2d visualization
    EMBEDDING_DIM = 2

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
    save_path = saver.save(sess, "w2v\\word_emb")
    print("Model saved in path: %s" % save_path)
    # Now the hidden layer (W1 + b1) is actually the word look up table
    vectors = sess.run(W1 + b1)
    print(vectors)

    w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
    w2v_df['word'] = words
    w2v_df = w2v_df[['word', 'x1', 'x2']]
    print('The output vector model is: \n',w2v_df)
