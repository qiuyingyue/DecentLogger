from sklearn import svm
from sklearn.cross_validation import train_test_split
import classifierHelp as help

def SVM():
    x, y = help.generateData()
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33, random_state = 42)

    ## use SVM to classify dataset
    # rbf is the default kernal: Gaussin kernal, ovr means one vs rest
    classifier=svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    classifier.fit(train_x, train_y)
    predictions=classifier.predict(test_x)
    help.evaluation(predictions, test_y)

def main():
    SVM()

if __name__ == '__main__':
    main()



    