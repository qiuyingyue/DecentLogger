import math
import os
import numpy as np
import pandas as pd
import dataProcess as dp

def evaluation(predictions, test_y):
    ## evaluation
    err_cnt=0
    for i in range(len(predictions)):
        if predictions[i]!= test_y[i]:
            err_cnt+=1
    print ('Accuracy', 1-err_cnt/float(len(predictions)))


def generateData():
    path = "clean_data"
    files = os.listdir(path)
    dfs = []
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(path + '/' + file, index_col=0)
            dfs.append(df)
    data = dp.preprocess(dfs)
    y = data['label'].values
    x = data.drop(df.columns[-1], axis=1).values
    return x, y