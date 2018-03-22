from sklearn import svm
from sklearn.cross_validation import train_test_split
import classifierHelp as help
from sklearn.externals import joblib

def SVM():
    #x, y = help.generateData()
    #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33, random_state = 42)
    
    train_x, test_x, train_y, test_y = help.generateTrainTest(preload=True)

    ## use SVM to classify dataset
    # poly is the best kernal for this case, ovr means one vs rest
    #classifier=svm.SVC(C=0.8, kernel='poly', gamma=20, decision_function_shape='ovr')
    #classifier.fit(train_x, train_y)
    #joblib.dump(classifier,'svm.pkl')
    classifier=joblib.load('svm.pkl')
    predictions=classifier.predict(test_x)
    help.evaluation(predictions, test_y)

def main():
    SVM()

if __name__ == '__main__':
    main()



    