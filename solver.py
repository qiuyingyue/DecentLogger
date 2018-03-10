#import NNsolver #finish later
#import SVMsolver #finish later
import os
import dataProcess as dp
import pandas as pd
#public interface
def train(dfs, method):
    if (method is "SVM"): #example
        df = dp.preprocess(dfs)
        print (df.head())
        #SVMsolver.train(df)
    
def test(method):
    pass




#Usage: only for unit testing
def main():
    path = "clean_data"
    files = os.listdir(path)
    dfs = []
    for file in files:
        if not os.path.isdir(path + '/' + file) and file.endswith(".csv"):
            df = pd.read_csv(path + '/' + file)
            dfs.append(df)
    print (len(dfs))
    df = dp.preprocess(dfs)
    print (len(df.index))
    # call preprocess function with needed parameters
    # def preprocess(dfs, method = "slides window", win_size = 5, step = 0.5)
    #train(dfs, "SVM")
main()
