from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
import classifierHelp as help

def LogisticClassifier():
    # x, y = help.generateData()
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.33, random_state = 42)
    
    train_x, test_x, train_y, test_y = help.generateTrainTest()

    ## use logistic regression to classify dataset
    classifier=LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs')
    classifier.fit(train_x, train_y)
    predictions=classifier.predict(test_x)
    help.evaluation(predictions, test_y)

def main():
    LogisticClassifier()

if __name__ == '__main__':
    main()



    