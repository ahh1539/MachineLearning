"""
@name Alexander Hurley
@project Iris Flower Identification Machine Learning
@date 12/4/2018
tutorial link: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
"""
import sys
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
# pandas
import pandas
# scikit-learn
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

#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"



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

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
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

def main():
    # This loads the dataset from the online line
    #version()
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    f = open("iris.csv")
    dataset = pandas.read_csv(f, names=names)
    statistics(dataset)
    machineLearning(dataset)

if __name__ == '__main__':
    main()
