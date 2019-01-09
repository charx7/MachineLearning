import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def embed_Dataframe(dataframeToEmbed, modelRoute):
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

        # get and evaluate tf variables for vocabulary and indexed vocabulary
        tf_vocabulary = restored_graph.get_tensor_by_name('vocabulary:0')
        vocab = tf_vocabulary.eval()
        tf_integerized_vocabulary = restored_graph.get_tensor_by_name('integerized_vocabulary:0')
        int_vocab = tf_integerized_vocabulary.eval()
        # strings need to be decoded
        # (tf thing: get rid of leading 'b' before each string)
        vocab = [w.decode() for w in vocab]

        # form embedding of sentence from its words' embeddings
        #ohv_test_sent = np.zeros((1,50)) # TODO restore embedding size don't hard code

        # For an embedding of the tweet
        # form a one-hot-vector of the test word
        ohv_test_word = np.zeros((1,sess.run('vocabulary_size:0')))

        # restore placeholders for input and operation
        test_input = restored_graph.get_tensor_by_name('one_hot_input:0')
        op_to_restore = restored_graph.get_tensor_by_name('w2v:0')

        # Create a numpy empty array
        embededData = []
        print('Embedding the Dataframe...')
        for tweet in tqdm(dataframeToEmbed):
            # For each word in the current tweet we run the embed
            ohv_test_sent = np.zeros((1,50)) # TODO restore embedding size don't hard code

            for word in tweet:
                # If we find it on the vocab (exists) then we use its embedding
                if word in vocab:
                    integerize_test_word = int_vocab[vocab.index(test_word)]
                    ohv_test_word[0, integerize_test_word] = 1;
                else:
                    ohv_test_word[0,0] = 1;

                # predict and print test word
                embed = sess.run(op_to_restore, {test_input:ohv_test_word})

                # Sum the embeded vectors to form the embeded sentence
                ohv_test_sent = ohv_test_sent + embed

            # Append to the data we are going to return
            embededData.append(ohv_test_sent)

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

        # get and evaluate tf variables for vocabulary and indexed vocabulary
        tf_vocabulary = restored_graph.get_tensor_by_name('vocabulary:0')
        vocab = tf_vocabulary.eval()
        tf_integerized_vocabulary = restored_graph.get_tensor_by_name('integerized_vocabulary:0')
        int_vocab = tf_integerized_vocabulary.eval()
        # strings need to be decoded
        # (tf thing: get rid of leading 'b' before each string)
        vocab = [w.decode() for w in vocab]

        # form embedding of sentence from its words' embeddings
        #ohv_test_sent = np.zeros((1,50)) # TODO restore embedding size don't hard code

        # For an embedding of the tweet
        # form a one-hot-vector of the test word
        ohv_test_word = np.zeros((1,sess.run('vocabulary_size:0')))
        # If we find it on the vocab (exists) then we use its embedding
        if word in vocab:
            integerize_test_word = int_vocab[vocab.index(test_word)]
            ohv_test_word[0, integerize_test_word] = 1;
        else:
            ohv_test_word[0,0] = 1;

        # restore placeholders for input and operation
        test_input = restored_graph.get_tensor_by_name('one_hot_input:0')
        op_to_restore = restored_graph.get_tensor_by_name('w2v:0')
        # predict and print test word
        prediction = sess.run(op_to_restore, {test_input:ohv_test_word})
        #print("The embedding of '{0}' is: {1}".format(test_word, prediction))
        sess.close()
    # Return the prediction
    return prediction

if __name__ == '__main__':
    # Test sentence
    test_sent = ['Oh', 'my', 'God', 'Taylor', 'Mascaras', 'on', 'sale!'] # TODO preprocess pipeline for test sentence

    # form embedding of sentence from its words' embeddings
    ohv_test_sent = np.zeros((1,50)) # TODO restore embedding size don't hard code
    for test_word in test_sent:
        # Call the embed funtion
        embed = embedTweet(test_word, 'tf_saved_models\\word_emb')
        # Sum the embeded vectors to form the embeded sentence
        ohv_test_sent = ohv_test_sent + embed

    # Print the vector result
    print("The embedding of '{0}' is: \n {1}".format(test_sent, ohv_test_sent))
