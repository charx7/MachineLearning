import pandas as pd
import numpy as np
import tensorflow as tf

# form test data
test_word = 'ray'
test_sent = ['make', 'feel', 'women', 'thoughtful', 'gifts'] # TODO preprocess pipeline for test sentence
# restore model
saver = tf.train.import_meta_graph("tf_saved_models\\word_emb.meta")
restored_graph = tf.get_default_graph()
with tf.Session(graph = restored_graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "tf_saved_models\\word_emb")
    # initialize restored variables
    sess.run(tf.global_variables())
    # print restored variables
    #print("Restored vocabulary size: {} \n".format(sess.run('vocabulary_size:0')))
    #print("Restored vocabulary: {} \n".format(sess.run('vocabulary:0')))
    #print("Restored integerized vocabulary: {} \n".format(sess.run('integerized_vocabulary:0')))
    #print("Restored weights: \n {} \n".format(sess.run('W1:0')))
    #print("Restored bias: {} \n".format(sess.run('b1:0')))
    
    # get and evaluate tf vriables for vocabulary and indexed vocabulary
    tf_vocabulary = restored_graph.get_tensor_by_name('vocabulary:0')
    vocab = tf_vocabulary.eval()
    tf_integerized_vocabulary = restored_graph.get_tensor_by_name('integerized_vocabulary:0')
    int_vocab = tf_integerized_vocabulary.eval()
    # strings need to be decoded
    # (tf thing: get rid of leading 'b' before each string)
    vocab = [w.decode() for w in vocab]

    # form embedding of sentence from its words' embeddings
    ohv_test_sent = np.zeros((1,50)) # TODO restore embedding size don't hard code
    for test_word in test_sent:
        # form a one-hot-vector of the test word
        ohv_test_word = np.zeros((1,sess.run('vocabulary_size:0')))
        if test_word in vocab:
            integerize_test_word = int_vocab[vocab.index(test_word)]
            ohv_test_word[0, integerize_test_word] = 1;
        else:        
            ohv_test_word[0,0] = 1;
        # restore placeholders for input and operation
        test_input = restored_graph.get_tensor_by_name('one_hot_input:0')
        op_to_restore = restored_graph.get_tensor_by_name('w2v:0')
        # predict and print test word
        prediction = sess.run(op_to_restore, {test_input:ohv_test_word})
        print("The embedding of '{0}' is: {1}".format(test_word, prediction))
        ohv_test_sent = ohv_test_sent + prediction
    print("The embedding of '{0}' is: \n {1}".format(test_sent, ohv_test_sent))
    
    
