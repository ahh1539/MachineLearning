
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

f = open("iris2.csv")
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'type','identifier']
flower_data = pd.read_csv(f, names=names)
y = flower_data.identifier
features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
X = flower_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

flower_model = DecisionTreeRegressor(random_state=1)

flower_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = flower_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
flower_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
flower_model.fit(train_X, train_y)
val_predictions = flower_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# from sklearn import tree
#
# features = [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5.0,3.6,1.4,0.2],
#             [5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],
#             [5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],
#             [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],[6.5,2.8,4.6,1.5],
#             [5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],
#             [5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],[6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],
#             [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3.0,5.8,2.2],
#             [7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],
#             [6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],[6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4]]
#
# labels = ["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
#           "2","2","2","2","2","2","2","2","2","2","2","2","2","2","2",
#           "3","3","3","3","3","3","3","3","3","3","3","3","3","3","3"]
#
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(features,labels)
# print(clf.predict([[5.1,3.8,1.6,0.2]]))