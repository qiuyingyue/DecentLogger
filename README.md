# DecentLogger

* Usage:
    * python main.py -extract [path]  #extract all the files in the directory specified in path
    * python main.py -process [path]  #process the files in the directory specified in path and output to the directory './all_data'
    * files can be splitted manually to two parts in './train_data' and './test_data'
    * python [classifier-name].py  #for classifier-name = logisticRegression, SVM, decisionTree, DNN, 
    * python [classifier-name].py -train #for training CNN and RNN (it costs long time)
    * python [classifier-name].py -test #for testing CNN and RNN with existing model
