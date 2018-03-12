from sklearn import svm
from sklearn import tree 
from sklearn.cross_validation import train_test_split
import classifierHelp as help

def decisionTree():
    x, y  = help.generateData()
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33)
    
    #train_x, test_x, train_y, test_y = help.generateTrainTest()

    ## use decisionTree to classify dataset
    # rbf is the default kernal: Gaussin kernal, ovr means one vs rest
    classifier = tree.DecisionTreeClassifier(criterion='gini')
    classifier.fit(train_x, train_y)
    predictions=classifier.predict(test_x)
    help.evaluation(predictions, test_y)

def main():
    decisionTree()

if __name__ == '__main__':
    main()



    