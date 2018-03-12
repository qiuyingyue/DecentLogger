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
        print (test_y[i])
    print ('Accuracy', 1-err_cnt/float(len(predictions)))


def generateData(preload = False, method = "slides window", win_size = 3, step = 0.2):
    path = "all_data"
    if preload and (os.path.exists(path + '/' + 'data.npy') and os.path.exists(path + '/' + 'labels.npy')):
        x = np.load(path + '/' + 'data.npy')
        y = np.load(path + '/' + 'labels.npy')
    else:
        files = os.listdir(path)
        dfs = []
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(path + '/' + file, index_col=0)
                dfs.append(df)
        x, y = dp.preprocess(dfs, method, win_size, step)
        np.save(path + '/' + "data", x)
        np.save(path + '/' + "labels", y)
    #y = data['label'].values
    #x = data.drop(df.columns[-1], axis=1).values
    return x, y

def generateTrainTest(preload = False, method = "slides window", win_size = 3, step = 0.2):
    path = "train_data"
    if preload and (os.path.exists(path + '/' + 'data.npy') and os.path.exists(path + '/' + 'labels.npy')):
        train_x = np.load(path + '/' + 'data.npy')
        train_y = np.load(path + '/' + 'labels.npy')
    else:
        files = os.listdir(path)
        dfs = []
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(path + '/' + file, index_col=0)
                dfs.append(df)
        train_x, train_y = dp.preprocess(dfs, method, win_size, step)
        np.save(path + '/' + "data", train_x)
        np.save(path + '/' + "labels", train_y)
    #train_y = train['label'].values
    #train_x = train.drop(df.columns[-1], axis=1).values
    path = "test_data"
    if preload and (os.path.exists(path + '/' + 'data.npy') and os.path.exists(path + '/' + 'labels.npy')):
        test_x = np.load(path + '/' + 'data.npy')
        test_y = np.load(path + '/' + 'labels.npy')
    else:
        files = os.listdir(path)
        dfs = []
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(path + '/' + file, index_col=0)
                dfs.append(df)
        test_x, test_y = dp.preprocess(dfs, method, win_size, step)
        np.save(path + '/' + "data", test_x)
        np.save(path + '/' + "labels", test_y)
    #test_y = test['label'].values
    #test_x = test.drop(df.columns[-1], axis=1).values
    return train_x, test_x, train_y, test_y
