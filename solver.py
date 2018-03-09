import NNsolver #finish later
import SVMsolver #finish later
from dataProcess import dp
import pandas as pd
#public interface
def train(df, method):
    if (method is "SVM"): #example
        SVMsolver.train(df)
def test(method):
    pass




#example: only for unit testing
def main():
    df1 = pd.read_csv("clean_data/xxx.csv")
    df2 = pd.read_csv("clean_data/xxx.csv")
    df = dp.preprocess([df1, df2])
    train(df)
main()
