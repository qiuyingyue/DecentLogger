import tensorflow as tf
import classifierHelp as help
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed

# hyperparameters
lr = 0.001                  # learning rate
max_steps = 2000            # train step upbound
batch_size = 100            
n_inputs = 23               # MNIST data input (img shape: 28*28)
n_steps = 300               # time steps
n_hidden_units = 500       # neurons in hidden layer
n_classes = 4              # MNIST classes (0-9 digits)


# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# initial weights biases
weights = {
    # shape (23, 500)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (500, 4)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # shape (500, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (4, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # X ==> (100 batches * 300 steps, 23 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (100 batches, 300 steps, 500 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # initial state as zero
    
    # if inputs is (batches, steps, inputs), time_major = false
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # final_state[1] is the h_state
    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # change outputs into [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']   
    
    return results

 # compute cost and train_op
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    train_x, test_x, train_y, test_y = help.generateTrainTest(preload = False, method="cnn")
    
    final_index = train_x.shape[0] - train_x.shape[0] % batch_size
    train_x = train_x[:final_index]
    train_y = train_y[:final_index]
    print(train_x.shape)
    train_x = train_x.reshape((-1, batch_size, n_steps, n_inputs))
    train_y = train_y.reshape((-1, batch_size))
    step = 0
    while step < max_steps:
        sess.run([train_op], feed_dict={
            x: train_x[step % train_x.shape[0]],
            y: train_y[step % train_y.shape[0]],
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: train_x[step % train_x.shape[0]],
            y: train_y[step % train_y.shape[0]],
            }))
        step += 1



