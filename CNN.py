import tensorflow as tf
import numpy as np 
import dataProcess as dp 
import os
import pandas as pd
import classifierHelp as helper
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    print("original:", features["x"].shape)
    shape = features["x"].shape
    input_layer = tf.reshape(features["x"], [-1, shape[1], shape[2], 1])
    print("input_layer:", input_layer)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=4,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print("conv1:", conv1)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print("pool1:", pool1)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=8,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print("conv2:", conv2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print("pool2:", pool2)
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1,   600])
    print("pool2_flat:", pool2_flat)
    dense = tf.layers.dense(inputs=pool2_flat, units=200, activation=tf.nn.relu)
    print("dense:", dense)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    print("dropout:", dropout)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=4)
    print("logits:", logits)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




# Create the Estimator
cnn_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="model/convnet_model")
# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
def train(train_data, train_labels):
    #train_data, train_labels = dp.preprocess(df, method = "dnn")
    print (train_data.shape, train_labels.shape)
     
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    cnn_classifier.train(
        input_fn=train_input_fn,
        steps=3000,
        hooks=[logging_hook])

def evaluate(eval_data, eval_labels):
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

def predict(data):
    pass

def main():
    train_data, test_data, train_labels, test_labels = helper.generateTrainTest(preload = False, method="cnn")
    
    '''path = "train_data"
    dfs = []
    for f in os.listdir(path):
        if f.endswith(".csv"):
            df = pd.read_csv(path + '/' + f, index_col = 0)
            dfs.append(df)
    print (len(dfs))
    train_data, train_labels = dp.preprocess(dfs, method="cnn")    
    path = "test_data"
    dfs = []
    for f in os.listdir(path):
        if f.endswith(".csv"):
            df = pd.read_csv(path + '/' + f, index_col = 0)
            dfs.append(df)
    print (len(dfs))
    test_data, test_labels = dp.preprocess(dfs, method="cnn")'''
    tf.logging.set_verbosity(tf.logging.INFO)
    
    train(train_data, train_labels)    
    evaluate(test_data, test_labels)
main()