import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt

from GLOBAL import label_dict
###########PUBLIC INTERFACE##############
#dfs: an array of dataframe
def preprocess(dfs, method = "slides window", win_size = 3, step = 0.2):
    if (method == "slides window"):
        dfs_new = []
        for df in dfs:
            df  = sliding_window(df, win_size = win_size, step = step)
            dfs_new.append(df)
        df_concat = pd.concat(dfs_new)
        return df_concat



###########PRIVATE INTERFACE################
def prepare_nn(df, win_size = 3, freq = 10, step = 0.8, withlabel = True):
    #drop label before concat
    if (withlabel):
        label = df.iloc[0].iloc[-1]
        df.drop(df.columns[-1], axis=1, inplace=True) 
    #calculate parameters
    total_rows = len(df.index)
    win_rows = win_size * (1000 / freq)
    print (type(df.columns))
    new_columns = df.columns * win_rows
    step_rows = win_rows * step

    for i in range(total_rows, step_rows):
        np_matrix = df[i, i + step_rows].values

    if (withlabel):
        pass
        
#unit of win_size is seconds
#unit of freq is ms
#step: the non-overlapping ratio of consecutive windows
def sliding_window(df, win_size = 3, freq = 10, step = 0.2, withlabel = True):

    #drop label before concat
    if (withlabel):
        label = df.iloc[0].iloc[-1]
        df.drop(df.columns[-1], axis=1, inplace=True)

    #calculate parameters
    total_rows = len(df.index)
    win_rows = int(win_size * (1000 / freq))
    #print ("win_rows",win_rows)
    new_columns = list(df.columns.values) * win_rows
    step_rows = int(win_rows * step)

    #create new dataframe
    np_rows = []
    #print ("total_rows:",total_rows)
    #print ("step_rows:",step_rows)
    
    for i in range(0, total_rows, step_rows):
        
        j = i + win_rows
        if (j >= total_rows):
            #print (i, i + win_rows)
            break
        np_row = [df.iloc[i: j].values.flatten()]
        #print(np_row[0].shape)
        np_rows.append(np_row)
    matrix = np.concatenate(np_rows)
    print (matrix.shape)
    df_new = pd.DataFrame(matrix, columns = new_columns)
    
    #add back label
    if (withlabel):
        df_new["label"] = pd.Series([label]*len(df_new.index), index = df_new.index) 

    return df_new
    #not tested yet

#change label from string to integer
def change_label(df, inplace = True):
    for label in label_dict:
        df.replace(to_replace=label, value=label_dict[label], inplace=inplace)
    return df
#change the format of the dataframe
def df_format(df, inplace = True, debug = False, withlabel = True):
    #delete unused column 
    if (withlabel):
        df.drop(df.columns[-2], axis=1, inplace=inplace)
    else:
        df.drop(df.columns[-1], axis=1, inplace=inplace)
        
    #reset index and column names
    new_columns = list(df.columns.values)
    new_columns[0] = "time"
    if (withlabel):
        new_columns[-1] = "label"
    df.columns = new_columns
    df.set_index("time", inplace=inplace)
    
    #truncate start and end
    start = int(df.index[0]) + 5 * 1000
    end = int(df.index[-1]) - 8 * 1000
    df = df[(df.index > start) & (df.index < end)]

    #change index type to DateTimeIndex
    df.index = pd.to_datetime(df.index, unit='ms')

   
    return df

#resample the data according to the frequency
#unit of freq: ms
def df_resample(df, freq = 10, inplace = True, debug = False):
    if (df.empty):
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


def df_draw(df):
    for col in df.columns:
        df.plot(x = df.index, y = col,title = col)
        #plt.show()
#test()
#fillData(df, start, end)

