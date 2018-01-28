import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

#data set from : https://en.wikipedia.org/wiki/Iris_flower_data_set?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt
def example_1():
    """
    This is just an example of how to use the data
    :return: void
    """

    #gets the data set of irises
    iris = load_iris()

    #feature catagories
    print (iris.feature_names)

    #types of iris names
    print(iris.target_names)

    #the first flower of data features
    print(iris.data[0])

    print(iris.target[0])

    for i in range(len(iris.target)):
        print("Example %d: lable %s, features %s" % (i,iris.target[i],iris.data[i]))
def example_2():
    iris = load_iris()

    #removing flower of each type
    test_idx = [0,50,100]

    #training dadta
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data,test_idx,axis = 0)


    #testing data
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    #train classifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data,train_target)

    #test if classifier predicts correctly
    print(test_target)
    print (clf.predict(test_data))

    #output:
    # [0 1 2]
    # [0 1 2]

    #create pdf graph to visuallize
    # from sklearn.externals.six import StringIO
    # import pydotplus
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                      feature_names=iris.feature_names,
    #                      class_names=iris.target_names,
    #                      filled=True,
    #                      rounded = True,
    #                      impurity = False)
    #
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("iris.pdf")

def example_3():
    """
    Using different classifier and getting the accuracy

    :return: void
    """
    from sklearn import datasets
    iris = datasets.load_iris()

    x = iris.data
    y = iris.target

    from sklearn.model_selection import train_test_split

    # split the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

    # Have the classifier learn
    # from sklearn import tree
    # my_classifier = tree.DecisionTreeClassifier()

    from sklearn.neighbors import KNeighborsClassifier
    my_classifier = KNeighborsClassifier()  # trying a different classifier

    my_classifier.fit(x_train, y_train)

    # run predictions
    predictions = my_classifier.predict(x_test)
    print(predictions)  # prints the type of iris predicted

    # see accuracy
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, predictions))

def example_4():
    """
    same as example_3 but with our own defined classifier "ScrappyKNN"
    :return:
    """
    """
       Using different classifier and getting the accuracy

       :return: void
       """
    from sklearn import datasets
    iris = datasets.load_iris()

    x = iris.data
    y = iris.target

    from sklearn.model_selection import train_test_split

    # split the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

    # Have the classifier learn
    # from sklearn import tree
    # my_classifier = tree.DecisionTreeClassifier()

    from sklearn.neighbors import KNeighborsClassifier
    my_classifier = ScrappyKNN()  # trying a different classifier

    my_classifier.fit(x_train, y_train)

    # run predictions
    predictions = my_classifier.predict(x_test)
    print(predictions)  # prints the type of iris predicted

    # see accuracy
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, predictions))

def euc(a,b):
    from scipy.spatial import distance
    return distance.euclidean(a,b)


import random
class ScrappyKNN():
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predictions = []

        for row in x_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self,row):
        best_dist = euc(row,self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row,self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]




example_4()
