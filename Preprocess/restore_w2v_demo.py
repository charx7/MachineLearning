import pandas as pd
import numpy as np
import tensorflow as tf

# form test data
test_one_hot_word = np.zeros((1,12), np.float32)
test_one_hot_word[0,0] = 1.0
print("One-hot array of test word (corresponds to 'girl'): \n {}".format(test_one_hot_word))
# restore and test model
saver = tf.train.import_meta_graph("w2v\\word_emb.meta")
restored_graph = tf.get_default_graph()
with tf.Session(graph = restored_graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "w2v\\word_emb")
    # initialize restored variables
    sess.run(tf.global_variables())
    # print restored weights and bias
    print("Restored weights: \n {} \n".format(sess.run('W1:0')))
    print("Restored bias: {} \n".format(sess.run('b1:0')))
    # restore placeholders for input and operation
    test_input = restored_graph.get_tensor_by_name('one_hot_input:0')
    op_to_restore = restored_graph.get_tensor_by_name("w2v:0")
    # predict and print test word
    prediction = sess.run(op_to_restore, {test_input:test_one_hot_word})
    print("The embedding of the test word is: \n {}".format(prediction))
