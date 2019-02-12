import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from preprocess import embedding_preprocess
from word2vec import preprocess_corpus

def embed_Dataframe(dataframeToEmbed, modelRoute):
    # Do the same preprocess as on the embed method
    dataframeToEmbed = preprocess_corpus(dataframeToEmbed)

    # Routes for restoration of our model
    saverRoute = modelRoute + '.meta'
    restoreRoute = modelRoute

    # Reset the graph
    tf.reset_default_graph()

    # Im not sure why do we need this one but it works XDXD
    # Restore the model we trained for word embeding
    restored_graph = tf.get_default_graph()

    # Start the tf session with the restored parameters to run the predict func
    with tf.Session(graph = restored_graph) as sess:

        # Import the graph
        saver = tf.train.import_meta_graph(saverRoute)

        saver = tf.train.Saver()
        saver.restore(sess, restoreRoute)
        # initialize restored variables
        sess.run(tf.global_variables())

        # get and evaluate tf variables for vocabulary and indexed vocabulary
        tf_vocabulary = restored_graph.get_tensor_by_name('vocabulary:0')
        vocab = tf_vocabulary.eval()
        tf_integerized_vocabulary = restored_graph.get_tensor_by_name('integerized_vocabulary:0')
        int_vocab = tf_integerized_vocabulary.eval()
        # strings need to be decoded
        # (tf thing: get rid of leading 'b' before each string)
        vocab = [w.decode() for w in vocab]

        # restore placeholders for input and operation
        test_input = restored_graph.get_tensor_by_name('x_pivot_idxs:0')
        op_to_restore = restored_graph.get_tensor_by_name('word_embed_lookup:0')

        # Create a numpy empty array
        embededData = []
        print('Embedding the Dataframe...')
        i = 0
        for tweet in tqdm(dataframeToEmbed):
            i = i + 1
            # For each word in the current tweet we run the embed
            emb_test_sent = np.zeros((1,sess.run('embedding_size:0')))
            # Init word counter as 0
            nwords = 0

            # Separate by whitespace
            words = tweet.split()
            for word in words:
                # If we find it on the vocab (exists) then we use its embedding
                if word in vocab:
                    # Get the number of Words
                    nwords = nwords + 1
                    integerize_test_word = int_vocab[vocab.index(word)]

                    # input for embedding look-up is an array of integerized words
                    int_words = []
                    int_words.append(integerize_test_word)
                    # predict and print test word
                    embed = sess.run(op_to_restore, {test_input:int_words})

                    #if (i % 100) == 0:
                        #print('current word is: ', word)
                        #print('The embed for the word: ', word, ' is: ', embed)

                else:
                    nwords = nwords + 1
                    # in case we didnt found it replace with the unk token
                    integerize_test_word = int_vocab[vocab.index('<UNK>')]

                    # input for embedding look-up is an array of integerized words
                    int_words = []
                    int_words.append(integerize_test_word)
                    # predict and print test word
                    embed = sess.run(op_to_restore, {test_input:int_words})

                    #if (i % 100) == 0:
                        #print('Not found word, the unk word is: ', word)

                        #print('The embed for the word: ', word, ' is: ', embed)
                    #integerize_test_word = 0
                    #embed = 0

                # Sum the embeded vectors to form the embeded sentence
                #emb_test_sent = emb_test_sent + embed
                emb_test_sent = np.add(emb_test_sent, embed)

            # Condition for no words founded
            if nwords == 0:
                # To prevent division by 0
                nwords = 1
                # To ensure we get the unknows token as the transformation
                integerize_test_word = int_vocab[vocab.index('<UNK>')]
                # input for embedding look-up is an array of integerized words
                int_words = []
                int_words.append(integerize_test_word)
                # predict and print test word
                emb_test_sent = sess.run(op_to_restore, {test_input:int_words})

            # Divide the result by number of words to get an average
            emb_test_sent = np.divide(emb_test_sent, nwords)

            # For debugz
            # if (i % 1000) == 0:
            #     # print for debug
            #     print('The number of words on the tweet is: ', nwords)
            #     print('The tweet: ', tweet, ' embed is: \n', emb_test_sent)

            # Append to the data we are going to return
            embededData.append(emb_test_sent)

    sess.close()

    npArrayEmbededData = np.array(embededData)
    reshapedNpArray = np.squeeze(npArrayEmbededData, axis=1)
    # Debug galore
    #print('The shape of the np array is: ',npArrayEmbededData.shape)
    #print('The generated np array is: ', npArrayEmbededData)
    #print('The shape of the squeezed np array is: ',reshaped.shape)
    #print('The generated squeezed np array is: ', reshaped)

    # Return the prediction
    return reshapedNpArray


def embedTweet(word, modelRoute):
    # Routes for restoration of our model
    saverRoute = modelRoute + '.meta'
    restoreRoute = modelRoute

    # Im not sure why do we need this one but it works XDXD
    # Restore the model we trained for word embeding
    restored_graph = tf.get_default_graph()

    # Reset the graph
    tf.reset_default_graph()
    # Start the tf session with the restored parameters to run the predict func
    with tf.Session(graph = restored_graph) as sess:

        # Import the graph
        saver = tf.train.import_meta_graph(saverRoute)

        saver = tf.train.Saver()
        saver.restore(sess, restoreRoute)
        # initialize restored variables
        sess.run(tf.global_variables())

        # restore placeholders for input and operation
        test_input = restored_graph.get_tensor_by_name('x_pivot_idxs:0')
        op_to_restore = restored_graph.get_tensor_by_name('word_embed_lookup:0')

        # get and evaluate tf variables for vocabulary and indexed vocabulary
        tf_vocabulary = restored_graph.get_tensor_by_name('vocabulary:0')
        vocab = tf_vocabulary.eval()
        tf_integerized_vocabulary = restored_graph.get_tensor_by_name('integerized_vocabulary:0')
        int_vocab = tf_integerized_vocabulary.eval()
        # strings need to be decoded
        # (tf thing: get rid of leading 'b' before each string)
        vocab = [w.decode() for w in vocab]

        # If we find it on the vocab (exists) then we use its embedding
        if word in vocab:
            integerize_test_word = int_vocab[vocab.index(test_word)]
        else:
            integerize_test_word = int_vocab[vocab.index('<UNK>')];

        # input for embedding look-up is an array of integerized words
        int_words = []
        int_words.append(integerize_test_word)

        # predict and print test word
        prediction = sess.run(op_to_restore, {test_input:int_words})
        #print("The embedding of '{0}' is: {1}".format(test_word, prediction))
        sess.close()
    # Return the prediction
    return prediction

if __name__ == '__main__':
    # Test sentence
    test_sent = ['Oh', 'my', 'God', 'Taylor', 'Mascaras', 'on', 'sale!'] # TODO preprocess pipeline for test sentence

    # form embedding of sentence from its words' embeddings
    emb_test_sent = np.zeros((1,128))
    for test_word in test_sent:
        # Call the embed funtion
        embed = embedTweet(test_word, 'tf_saved_models\\word_emb')
        # Sum the embeded vectors to form the embeded sentence
        emb_test_sent = emb_test_sent + embed

    # Print the vector result
    print("The embedding of '{0}' is: \n {1}".format(test_sent, emb_test_sent))
