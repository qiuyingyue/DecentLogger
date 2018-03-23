import sys
import fileProcess as fp 
import pandas as pd
import CNN
def main(argv):
    # My code here
    print(argv)
    
    if (argv[1] == "-extract"):
        dataPath = argv[2] 
        fp.extract_files(dataPath)
    elif (argv[1] == "-process"):
        dataPath = argv[2]
        fp.get_dataframes(dataOutRoot = "all_data/", debug = False)
    else:
        print("Please specify extract/process phase and data path")
        sys.exit(0)
    


if __name__ == '__main__':
    main(sys.argv)