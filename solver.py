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
    elif (method == "DNN"):
        train_data, train_label = dp.preprocess(dfs, method="dnn")
        CNN.train(train_data, train_label)
        
    
def test(dfs, method):
    if (method == "SVM"): #example
        df = dp.preprocess(dfs, "slides window")
        print(df.head())
    elif (method == "DNN"):
        test_data, test_label = dp.preprocess(dfs, method="dnn")
        CNN.evaluate(test_data, test_label)




#Usage: only for unit testing
def main():
    path = "clean_data"
    dfs = []
    for f in os.listdir(path):
        if f.endswith(".csv"):
            df = pd.read_csv(path + '/' + f, index_col = 0)
            dfs.append(df)
    print (len(dfs))
    train(dfs, "DNN") 
    path = "test_data"
    dfs = []
    for f in os.listdir(path):
        if f.endswith(".csv"):
            df = pd.read_csv(path + '/' + f, index_col = 0)
            dfs.append(df)
    print (len(dfs))

    test(dfs, "DNN")   
    
    # call preprocess function with needed parameters
    # def preprocess(dfs, method = "slides window", win_size = 5, step = 0.5)
    #train(dfs, "SVM")
main()
