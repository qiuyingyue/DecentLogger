import os
import shutil
import gzip
import pandas as pd
import numpy as np 
import dataProcess as dp
import time
from GLOBAL import sensors, labels
#extract the data csv file from .gz
def extract_files(dataInRoot = "../Sessions"):
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
def get_dataframe(dataInRoot, dataOutRoot = "clean_data/", debug = False, withlabel = True):
    df_dict = {}
    label = ""
    if not (os.path.exists(dataOutRoot)):
        os.makedirs(dataOutRoot)
    for fname in os.listdir(dataInRoot):
        if (not (fname.endswith(".csv"))) or ("65539" in fname):
            continue
        print (fname)
        #read file
        df = pd.read_csv(os.path.join(dataInRoot,fname), header=None)
        #get label and sensor before process
        label = df.iloc[0].iloc[-1]
        sensor = fname.split('.')[-3]
        
        if (sensor not in sensors):
            continue

        
        #process file
        df = dp.df_format(df, sensor, debug = debug)
        
        df = dp.df_resample(df, debug = debug)
        
        #Debug only: write to files
        for s in sensors:
            if (s in fname) and debug:
                print (label, s)
                fname = os.path.join(dataOutRoot,"{}.resample.{}.{}.data.csv".format(int(time.time()), label, s))
                df.to_csv(fname)
                print("written to ", fname)

        #drop label before concat
        if (withlabel):
            df.drop(df.columns[-1], axis=1, inplace=True)

        
        df_dict[sensor] = df
        
    #ensure the order of sensor files  
    print("sensors", sensors)

    print("df_dict", df_dict.keys())  
    dfs = []
    for sensor in sensors:
        if (sensor in df_dict and sensor != "step_counter"):
            dfs.append(df_dict[sensor])
    #merge files with different sensors into one file
    df_concat = pd.concat(dfs, axis=1, join='inner')
    df_concat = df_concat.join(df_dict["step_counter"], how="left") 
    #print ("Before",df_concat.head())
    df_concat.interpolate(method='time',  axis=0, inplace=True)
    df_concat = df_concat.fillna(0)
    #print ("After",df_concat.head())

    #add back label
    if (withlabel):
        df_concat["label"] = pd.Series([label]*len(df_concat.index), index = df_concat.index) 
    
    #write to files
    print(label, len(dfs))
    dp.change_label(df_concat)    
    fname = "{}.resample.{}.data.csv".format(int(time.time()), label)
    df_concat.to_csv(os.path.join(dataOutRoot, fname))
    return df_concat, label

def get_dataframes(dataInRoot = "../Sessions", dataOutRoot = "clean_data/", debug = False):
    #read files and truncate its head and tail
    for root, dirs, files in os.walk(dataInRoot):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        df_list = []
        if (os.path.basename(root) == "data"):
            df, label = get_dataframe(root, dataOutRoot, debug)
            df_list.append(df)
    return df_list
        
                

#For unit testing                
#extractFiles()
#get_dataframes(dataOutRoot = "all_data/", debug = False)
#getDataframe("../Sessions/14442D5DF8A8DC4_Mon_Feb_26_13-37_2018_PST/data",debug = True)
