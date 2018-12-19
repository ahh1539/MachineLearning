
"""
@name Alexander Hurley
@project Kaggle techniques
@date 12/5/2018
"""
from machine_learning import datasets
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

#leaf nodes are breaks in data sets
# Using best value for max_leaf_nodes
flower_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
flower_model.fit(train_X, train_y)
val_predictions = flower_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

