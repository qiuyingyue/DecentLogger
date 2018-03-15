import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
import sys
from GLOBAL import label_dict, sensors
###########PUBLIC INTERFACE##############
#dfs: an array of dataframe
#norm: whether to normalize
#method: 
#uwin_size: size of sliding window, unit is seconds
#unit of freq is ms
#step: the non-overlapping ratio of consecutive windows
def preprocess(dfs, norm, method, win_size, step, withlabel = True):
    '''if (method == "slides window"):
        dfs_new = []
        for df in dfs:
            if (withlabel):
                labels = df['label'].values
                df.drop(df.columns[-1], axis=1, inplace=True)
                list_labels.append(labels)
            if (norm):
                df = normalize(df)
            df  = sliding_window(df, win_size = win_size, step = step)
            dfs_new.append(df)
        df_concat = pd.concat(dfs_new)
        y = df_concat['label'].values
        x = df_concat.drop(df_concat.columns[-1], axis=1).values
        print ("total shape:", df_concat.values.shape)'''
    
    list_data = []
    list_labels = []
    for df in dfs:
        if (withlabel):
            label = df['label'].values[0]
            df.drop(df.columns[-1], axis=1, inplace=True)
        if (norm):
            print("normalize")
            df = normalize(df)
            
        if (method == "slides window"):
            data = sliding_window(df, win_size=win_size, step = step)
        elif (method == "3d"):
            data = prepare_cnn(df, win_size=win_size, step=step)
        else:
            print("Please specify correct method name: 'slides window' or '3d'")
            sys.exit(0)
        list_data.append(data)
        if (withlabel):
            list_labels.append(np.full(data.shape[0], label, int))
    
    x = np.concatenate(list_data)
    print("final data dimention:", x.shape)
    if (withlabel):
        y = np.concatenate(list_labels)
        #print ("x,y,", x.shape, y.shape)
        return x, y
    else:
        return x

def normalize(df):
    np_array = df.values
    '''np_max = np.amax(np_array, axis=0)  
    np_min = np.amin(np_array, axis=0)
    np_diff = np_max - np_min
    print(np_diff[4])'''
    np_diff = np.array([156.96, 156.96, 156.96, #accelerator: 156.96
    4915, 4915, 4915, #magnetic 4915.
    360, 360, 360,   #orientation 360
    34.9, 34.9, 34.9, #gyo: 34.9
    19.6133, 19.6133, 19.6133, #gravity: 19.6133
    19.6133, 19.6133, 19.6133, #linear aceleration: 19.6133
    1, 1, 1, 1, 1, #rotation vector: 1   (5)
    1])#step counter
    np_array = np_array/np_diff
    np_array[:,-1] = np_array[:,-1]-np_array[0][-1]
    #print(np_array)
    new_df= pd.DataFrame(np_array, index = df.index, columns = df.columns)
    return new_df
        
    

###########PRIVATE INTERFACE################
def prepare_cnn(df, win_size, step, freq = 10):
    #calculate parameters
    total_rows = len(df.index)
    win_rows = int(win_size * (1000 / freq))
    step_rows = int(win_rows * step)
    #print ("win_rows",win_rows)
    #print ("total_rows:",total_rows)
    #print ("step_rows:",step_rows)
    np_matrixs = []
    for i in range(0, total_rows, step_rows):
        if (i + win_rows >= total_rows):
            break
        np_matrix = df[i: i + win_rows].values
        
        np_matrixs.append(np_matrix)
    train_data = np.stack(np_matrixs)
    #print("in prepare cnn",train_data.shape)
    train_data =  train_data.astype(np.float32)
    return train_data
    
def sliding_window(df, win_size = 3, step = 0.2, freq = 10):
    #calculate parameters
    total_rows = len(df.index)
    win_rows = int(win_size * (1000 / freq))
    step_rows = int(win_rows * step)
    #print ("win_rows",win_rows)
    #print ("total_rows:",total_rows)
    #print ("step_rows:",step_rows)
    
    #new_columns = list(df.columns.values) * win_rows
    
    #create new data matrix
    np_rows = []
    for i in range(0, total_rows, step_rows):
        j = i + win_rows
        if (j >= total_rows):
            break
        np_row = df.iloc[i: j].values.flatten()
        np_rows.append(np_row)
    data_matrix = np.stack(np_rows)
    #print ("shape:",data_matrix.shape)
    return data_matrix

#change label from string to integer
def change_label(df, inplace = True):
    for label in label_dict:
        df.replace(to_replace=label, value=label_dict[label], inplace=inplace)
    return df


#change the format of the dataframe
def df_format(df, sensor, truncate = True, inplace = True, debug = False, withlabel = True):
    
    #delete unused column 
    if (withlabel):
        df.drop(df.columns[-2], axis=1, inplace=inplace)
    else:
        df.drop(df.columns[-1], axis=1, inplace=inplace)
        
    #reset index and column names
    new_columns = list(df.columns.values)
    new_columns[0] = "time"
    for i in range(1, len(new_columns)):
        new_columns[i] = sensor + str(new_columns[i])
    if (withlabel):
        new_columns[-1] = "label"
    df.columns = new_columns
    df.set_index("time", inplace=inplace)
    
    #truncate start and end
    if (truncate):
        start = int(df.index[0]) + 8 * 1000
        end = int(df.index[-1]) - 10 * 1000
        df = df[(df.index > start) & (df.index < end)]

    #change index type to DateTimeIndex
    df.index = pd.to_datetime(df.index, unit='ms')

   
    return df

#resample the data according to the frequency
#unit of freq: ms
def df_resample(df, freq = 10, inplace = True, debug = False):
    if (df.empty):
        print (df)
        return df
    if (debug):
        print("\nBefore resample:\n", df.iloc[0:10])
        df.to_csv("original.csv")
    #get label
    label = df.iloc[0].iloc[-1]

    #populate the dataframe with time frequence of 10 ms and fill them with nan
    df = df[~df.index.duplicated(keep='first')] # remove duplicated index
    dfnan = df.resample(str(freq)+'ms').asfreq()
    dfnan.drop(df.columns, axis=1, inplace=inplace)
    df = dfnan.join(df, how = "outer")

    #debug info    
    if (debug):
        print("\nAfter resample:\n", df.iloc[0:10])
        df.to_csv("resample.csv")

    #interpolate to fill nan with values linear to time interval
    t_start = time.clock()
    df.interpolate(method='time',  axis=0, inplace=inplace)

    #downsample to 10ms (delete the time index not divisible by 10 ms)
    #drop first row and last row which cannot be interpoated
    df = df.resample('10ms').asfreq()
    df.drop(df.index[0], inplace=inplace)
    df.drop(df.index[-1], inplace=inplace)

    #fill labels
    df.fillna(label, inplace=inplace)

    #debug info
    if (debug):
        print("\nFinish interpolation:\n")
        print(df.iloc[0:10])
        print("Finish interpolation:", time.clock() - t_start, "seconds")
        df.to_csv("interpolation.csv")
    return df




