import tensorflow as tf
import classifierHelp as help
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import train_test_split
tf.set_random_seed(1)   # set random seed

# hyperparameters
lr = 0.001                  # learning rate
max_steps = 20000            # train step upbound
batch_size = 100            
n_inputs = 24               # input demensions
n_steps = 100               # time steps
n_hidden_units = 30       # neurons in hidden layer
n_classes = 4               


# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int64,[None, 1])

# initial weights biases
weights = {
    # shape (23, 500)
    'in': tf.get_variable(name='in', shape=(n_inputs, n_hidden_units), initializer=tf.orthogonal_initializer(),
                dtype=tf.float32),
    # shape (500, 4)
    'out': tf.get_variable(name='out', shape=(n_hidden_units, n_classes), initializer=tf.orthogonal_initializer(),
                dtype=tf.float32)
}

biases = {
    # shape (500, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (4, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
} 

def RNN(X, weights, biases):
    # X ==> (100 batches * 60 steps, 23 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (100 batches, 60 steps, 500 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=0.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(tf.shape(X_in)[0], dtype=tf.float32) # initial state as zero
    
    # if inputs is (batches, steps, inputs), time_major = false
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # final_state[1] is the h_state
    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # change outputs into [(batch, outputs)..] * steps
    #if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    #else:
    #    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    #results = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
    
    return results

# compute cost and train_op
pred = RNN(x, weights, biases)
cost = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=pred)
train_op = tf.train.AdamOptimizer(lr).minimize(loss=cost, global_step=tf.train.get_global_step())
#train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss=cost, global_step=tf.train.get_global_step())

correct_pred = tf.equal(tf.argmax(pred, 1), tf.reshape(y, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

outputlabel = tf.argmax(pred, 1)
#outputlabel = y

with tf.Session() as sess:
    
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)


    saver = tf.train.Saver()
    if os.path.exists('./model/rnn/'):
        model_file=tf.train.latest_checkpoint('./model/rnn/')
        saver.restore(sess, model_file)
    

    train_x, test_x, train_y, test_y = help.generateTrainTest(preload=False, win_size=1, method="3d")
    #data_x, data_y = help.generateData(preload=False, win_size=0.2, method="3d")
    #train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.2, random_state = 42)
    
    final_index = train_x.shape[0] - train_x.shape[0] % batch_size
    train_x = train_x[:final_index]
    train_y = train_y[:final_index]
  
    train_x = train_x.reshape((-1, batch_size, n_steps, n_inputs))
    train_y = train_y.reshape((-1, batch_size, 1))
    
    test_y = test_y.reshape((-1, 1))

    step = 0
    

    while step < max_steps:
        '''sess.run([train_op], feed_dict={
            x: train_x[step % train_x.shape[0]],
            y: train_y[step % train_y.shape[0]],
        })'''
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: test_x,
            y: test_y,
            })) 
            help.evaluation(sess.run(outputlabel,  feed_dict={
            x: test_x,
            y: test_y,
            }), test_y.reshape(-1))
            
        step += 1
        if step % 500 == 0:
            saver.save(sess, './model/rnn/my.ckpt', global_step=tf.train.get_global_step())
    



