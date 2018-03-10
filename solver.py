#import NNsolver #finish later
#import SVMsolver #finish later
import os
import dataProcess as dp
import pandas as pd
import CNN 
#public interface
def train(dfs, method):
    if (method == "SVM"): #example
        df = dp.preprocess(dfs, "slides window")
        print(df.head())
    elif (method == "CNN"):
        train_data, train_labels = dp.preprocess(dfs, method="cnn")
        CNN.train(train_data, train_labels)
        
    
def test(dfs, method):
    if (method == "SVM"): #example
        df = dp.preprocess(dfs, "slides window")
        print(df.head())
    elif (method == "CNN"):
        test_data, test_labels = dp.preprocess(dfs, method="cnn")
        CNN.evaluate(test_data, test_labels)




#Usage: only for unit testing

