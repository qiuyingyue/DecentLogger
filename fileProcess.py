import os
import shutil
import gzip
import pandas as pd
import numpy as np 
import dataProcess as dp
sensors = ["accelerometer", "magnetic_field", "orientation", 
                "gyroscope", "light", "gravity",
                "linear_acceleration", "rotation_vector", "step_counter"]
labels = ["sitting", "standing", "walking", "laying_down"]
#extract the data csv file from .gz
def extractFiles(dataInRoot = "../Sessions"):
    print ("Extracting", dataInRoot)
    for root, dirs, files in os.walk(dataInRoot):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        for fname in files:
            print( len(path) * '---', fname)
            if (fname.endswith('.gz')):
                print ("extracting" )
                with gzip.open(os.path.join(root,fname), 'rb') as f_in:
                    newfname = fname[:-3]
                    with open(os.path.join(root,newfname), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        print ('extract', fname, 'to' ,newfname)
                        os.remove(os.path.join(root,fname)) 
def getDataframe(dataPath, debug = False):
    dfs = []
    label = ""
    dirname = os.path.join(dataPath,"../clean_data")
    if not (os.path.exists(dirname)):
        os.makedirs(dirname)
    for fname in os.listdir(dataPath):
        if (not (fname.endswith(".csv"))) or ("65539" in fname):
            continue
        print (fname)
        #read file
        df = pd.read_csv(os.path.join(dataPath,fname), header=None)
        #get label before process
        label = df.iloc[0].iloc[-1]
        #process file
        df = dp.processSingle(df, debug = debug)
        if (df.empty):
            continue
        
        #write to files
        for sensor in sensors:
            if (sensor in fname) and debug:
                print (label, sensor)
                fname = os.path.join(dirname,"reconstructed.{}.{}.data.csv".format(label, sensor))
                df.to_csv(fname)
                print("written to ", fname)
        df.drop(df.columns[-1], axis=1, inplace=True)#drop label before concat
        dfs.append(df)
    #merge files with different sensors into one file
    df_concat = pd.concat(dfs, axis=1, join='inner') 
    df_concat["label"] = pd.Series([label]*len(df_concat.index), index = df_concat.index) 
    print(label, len(dfs))
    fname = "reconstructed.{}.data.csv".format(label)
    
    df_concat.to_csv(os.path.join(dirname,"../clean_data",fname))
    dp.changeLabel(df_concat)
    return df_concat, label
#reorder files according to their labels
'''def reorderFiles(dataInRoot = "../Sessions", dataOutRoot = "../clean_data/"):
    # create the write data directory
    dataframes = {}
    for label in labels:
        dataframes[label]={}
        for sensor in sensors:
            dataframes[label][f]=[]
            dirname = os.path.join(dataOutRoot,label,f)
            #if (os.path.exists(dirname)):
            #   os.remove(dirname)
            if not (os.path.exists(dirname)):
                os.makedirs(dirname)'''

def getDataframes(dataInRoot = "../Sessions", dataOutRoot = "../clean_data/", debug = False):
    #read files and truncate its head and tail
    for root, dirs, files in os.walk(dataInRoot):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        df_list = []
        if (os.path.basename(root) == "data"):
            df, label = getDataframe(root, debug)
            df_list.append(df)
    return df_list
    '''if (label in df_dict):
        df_dict[label]=[]
    df_dict[label].append(df)'''
    '''df_total = []
    for label in labels:
        df_concat = pd.concat(df_dict[label])
        df_total.append(df_concat)
    return pd.concat(df_total)'''

        
                

                 
#extractFiles()
getDataframes(debug = False)
#getDataframe("../Sessions/14442D5DF8A8DC4_Mon_Feb_26_13-37_2018_PST/data",debug = True)
