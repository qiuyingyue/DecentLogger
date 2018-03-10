import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import classifierHelp as help



def DNN():
    # get data
    
    # x, y = help.generateData()
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33, random_state = 42)
    
    train_x, test_x, train_y, test_y = help.generateTrainTest()

    train_y = train_y.astype(int)
    test_y = test_y.astype(int)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension = 6900)]
    # three 1000-units hidden layers
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[500, 500, 500], n_classes=4)
    classifier.fit(x=train_x, y=train_y, steps=2000, batch_size=100)
    predictions = list(classifier.predict(test_x, as_iterable=True))
    help.evaluation(predictions, test_y)


def main():
    DNN()

if __name__ == '__main__':
    main()




