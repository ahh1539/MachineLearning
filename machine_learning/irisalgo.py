"""
@name Alexander Hurley
@project Iris Flower Identification Machine Learning
@date 12/4/2018
tutorial link: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
"""
import sys
import scipy
import numpy
import matplotlib
import sklearn

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def version():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))



def statistics(dataset):
    # shape
    print(dataset.shape)

    # shape, (rows and colums)
    print(dataset.shape)

    #Prints out the different classifications of the dataset
    # class distribution
    print(dataset.groupby('class').size())

    # Prints our first "x" rows of data
    x = 20
    print(dataset.head(x))

    # Statistical analysis of data
    print(dataset.describe())

def graphs(dataset):
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
    # Scatterplot
    scatter_matrix(dataset)
    plt.show()
    # Histogram
    dataset.hist()
    plt.show()



"""
Training data set 80% used to train, 20% used to test accuracy
"""
def machineLearning(dataset):
    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = \
        model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'
    # Spot Check Algorithms
    print("ML MODELS:")
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)



def main():
    # This loads the dataset from the online line
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    f = open("iris.csv")
    dataset = pandas.read_csv(f, names=names)
    #version()
    #graphs(dataset)
    statistics(dataset)
    machineLearning(dataset)

if __name__ == '__main__':
    main()
