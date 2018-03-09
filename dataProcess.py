import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt

def processSingle(df):
    df = setIndex(df)
    df = resample(df)
    return df

def setIndex(df, debug = False):
    #delete unused column 
    df.drop(df.columns[-2], axis=1, inplace=True)
    #reassisn index and column names
    newColNames = list(df.columns.values)
    newColNames[0] = "time"
    newColNames[-1] = "label"
    df.columns = newColNames
    #print(df.head())
    df.set_index("time", inplace=True)
    #truncate start and end
    newStart = int(df.index[0]) + 5 * 1000
    newEnd = int(df.index[-1]) - 8 * 1000
    df = df[(df.index > newStart) & (df.index < newEnd)]
    #change index type to DateTimeIndex
    df.index = pd.to_datetime(df.index, unit='ms')
    #check result
    if (debug):
        print(df.head())
    return df
def resample(df, interpolate = True, freq = 10, inplace = True, debug = False):
    if (df.empty):
        return df
    label = df.iloc[0].iloc[-1]
    #init_time = df.index[0]
    dfnan = df.resample(str(freq)+'ms').asfreq()
    dfnan.drop(df.columns, axis=1, inplace=inplace)
    df = dfnan.join(df, how = "outer")
    if (debug):
        print("\nAfter resample:\n",df.iloc[0:10])
        df.to_csv("test.csv")
    #interpolate
    if (interpolate):
        
        t_start = time.clock()
        df.interpolate(method='time',  axis=0, inplace=inplace)
        #df.fillna(label, inplace=inplace)
        df = df.resample('10ms').asfreq()
        df.drop(df.index[0], inplace=inplace)
        df.drop(df.index[-1], inplace=inplace)
        if (debug):
            print("\Finish interpolation:\n")
            print(df.iloc[0:10])
            print("Finish interpolation:", time.clock() - t_start)
            df.to_csv("test2.csv")
            
    return df


def draw(df):
    for col in df.columns:
        df.plot(x = df.index, y = col,title = col)
        plt.show()
#test()
#fillData(df, start, end)

