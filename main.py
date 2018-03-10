import sys
import fileProcess as fp 
import solver
import pandas as pd
def main(argv):
    # My code here
    print(argv)
    
    if (argv[1] == "-train"):
        traindataPath = argv[2]
        dfsTrain = fp.get_dataframes(traindataPath) 
        solver.train(dfsTrain, method = "DNN") 
    elif (argv[1] == "-test"):
        testdataPath = argv[2]
        dfsTest = fp.get_dataframes(testdataPath)
        solver.test(dfsTest)
    elif (argv[1] == "-extract"):
        dataPath = argv[2] 
        fp.extract_files(dataPath)
    else:
        print("Please specify train/test phase and data path")
        sys.exit(0)
    


if __name__ == '__main__':
    main(sys.argv)