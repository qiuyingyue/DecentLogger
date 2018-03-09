import sys
import fileProcess as fp 
#import XXsolver 
import solver #example
#import SVMsolver
import pandas as pd
def main(argv):
    # My code here
    if (len(argv))
    if (argv[1] is "train"):
        if (len(argv) > 3 and argc[3] is "-x"):
            fp.extract_files(dataPath)
        traindataPath = argv[2]
        dfsTrain = fp.get_dataframes(dataPath) 
        dfTrain = dp.preprocess(dfsTrain)
        solver.train(dfsTrain) 
    elif (argv[1] is "test"):
        testdataPath = argv[2]
        dfTest = fp.get_dataframe(dataPath)
        solver.test(dfTest)
    else:
        print("Please specify train/test phase and data path")
        sys.exit(0)
    


if __name__ == '__main__':
    main(sys.argv)