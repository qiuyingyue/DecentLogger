import math
import os
import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
import dataProcess as dp

path = "clean_data"
files = os.listdir(path)

def evaluation(predictions, test_y):
    ## evaluation
    err_cnt=0
    for i in range(len(predictions)):
        if predictions[i]!= test_y[i]:
            err_cnt+=1
    print ('Accuracy', 1-err_cnt/float(len(predictions)))


def LogisticClassifier():
    dfs = []
    for file in files:
        if not os.path.isdir(path + '/' + file):
            df = pd.read_csv(path + '/' + file)
            dfs.append(df)
    data = dp.preprocess(dfs)
    print (data)
    y = data['label']
    x = data.drop(label)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33, random_state = 42)
    
    ## use logistic regression to classify dataset
    classifier=LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs')
    classifier.fit(train_x, train_y)
    predictions=classifier.predict(test_x)
    evaluation(predictions, test_y)

def main():
    LogisticClassifier()

if __name__ == '__main__':
    main()



    