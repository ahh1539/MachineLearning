"""
@name Alexander Hurley
@project Iris Flower Identification Machine Learning
@date 12/5/2018
"""
import sklearn
import numpy as np

from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

print(iris.feature_names)
print(iris.data[0])
print(iris.target_names)
# label 0 == setosa, label 1 == versicolor, label 2 == virginica
print(iris.target[0])

test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)



#print("Actual Results Numerical:", test_target)
actual = ""
for item in test_target:
    if item == 0:
        actual += "setosa, "
    elif item == 1:
        actual += "versicolor, "
    else:
        actual += "virginica, "
print("Actual Results:", actual)

#print("Predicted Results Numerical:", clf.predict(test_data))
predicted = ""
for item in test_target:
    if item == 0:
        predicted += "setosa, "
    elif item == 1:
        predicted += "versicolor, "
    else:
        predicted += "virginica, "
print("Predicted Results:", predicted)

