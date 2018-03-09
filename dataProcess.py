import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
label_dict = {"sitting": 0, "standing": 1, "walking": 2, "laying_down": 3}

def preprocess(dfs, method = "slides window"):
    if (method is "slides window"):
        pass
def processSingle(df, withlabel = True, debug = False):
    df = formatDataframe(df, debug = debug)
    df = resample(df, debug = debug)
    return df
#change label from string to integer
def changeLabel(df, inplace = True):
    for label in label_dict:
        df.replace(to_replace=label, value=label_dict[label], inplace=inplace)
    return df
#change the format of the dataframe
def formatDataframe(df, inplace = True, debug = False, withlabel = True):
    #delete unused column 
    if (withlabel):
        df.drop(df.columns[-2], axis=1, inplace=inplace)
    else:
        df.drop(df.columns[-1], axis=1, inplace=inplace)
        
    #reset index and column names
    newColNames = list(df.columns.values)
    newColNames[0] = "time"
    if (withlabel):
        newColNames[-1] = "label"
    df.columns = newColNames
    df.set_index("time", inplace=inplace)
    
    #truncate start and end
    newStart = int(df.index[0]) + 5 * 1000
    newEnd = int(df.index[-1]) - 8 * 1000
    df = df[(df.index > newStart) & (df.index < newEnd)]

    #change index type to DateTimeIndex
    df.index = pd.to_datetime(df.index, unit='ms')

   
    return df

#resample the data according to the frequency
#unit of freq: ms
def resample(df, freq = 10, inplace = True, debug = False):
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
        df.to_csv("test.csv")

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
        df.to_csv("test2.csv")
    return df


def draw(df):
    for col in df.columns:
        df.plot(x = df.index, y = col,title = col)
        plt.show()
#test()
#fillData(df, start, end)

