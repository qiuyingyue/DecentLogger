import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt

from GLOBAL import label_dict

def preprocess(dfs, method = "slides window"):
    if (method is "slides window"):
        dfs = []
        for df in dfs:
            df  = sliding_window(df)
            dfs.append(df)
        df_concat = pd.concat(dfs)


#unit of win_size is seconds
#unit of freq is ms
#step is the overlapping ratio of consecutive windows
def sliding_window(df, win_size = 5, freq = 10, step = 0.5, withlabel = True):
    label = df.iloc[0].iloc[-1]

    #drop label before concat
    if (withlabel):
        df.drop(df.columns[-1], axis=1, inplace=True)

    #calculate parameters
    total_rows = len(df.index)
    win_rows = win_size * (1000 / freq)
    new_columns = df.columns * win_rows
    step_rows = win_rows * step

    #create new dataframe
    np_rows = []
    for i in range(total_rows, step_rows):
        np_row = [df[i, i + step_rows].values.flatten()]
        np_rows.append(np_row)
    matrix = np.concatenate(np_rows)
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
        print("\Before resample:\n", df.iloc[0:10])
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
        print("\Finish interpolation:\n")
        print(df.iloc[0:10])
        print("Finish interpolation:", time.clock() - t_start, "seconds")
        df.to_csv("interpolation.csv")
    return df


def df_draw(df):
    for col in df.columns:
        df.plot(x = df.index, y = col,title = col)
        plt.show()
#test()
#fillData(df, start, end)

