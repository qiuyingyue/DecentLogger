from sklearn import svm
from sklearn import tree 
from sklearn.cross_validation import train_test_split
import classifierHelp as help
from sklearn.externals import joblib

def decisionTree():
    #x, y  = help.generateData()
    #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33)
    
    train_x, test_x, train_y, test_y = help.generateTrainTest(preload=False, win_size=8)

    ## use decisionTree to classify dataset
    
    classifier = tree.DecisionTreeClassifier(criterion='gini')
    classifier.fit(train_x, train_y)
    joblib.dump(classifier,'decisionTree.pkl')
    classifier=joblib.load('decisionTree.pkl')
    predictions=classifier.predict(test_x)
    help.evaluation(predictions, test_y)

def main():
    decisionTree()

if __name__ == '__main__':
    main()



    