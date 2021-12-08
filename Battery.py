import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Preparing data
names = ['Charging Time', 'Battery Live']
df = pd.read_csv('trainingdata.txt', names=names)
X = [[int(value*10)] for value in df[names[0]]]
y = [int(value*10) for value in df[names[1]]]

# Choosing classifier and adjusting by GridSearchCV
clf = RandomForestRegressor()
param_grid = [
    {'n_estimators': [3, 10, 30]},
    {'bootstrap': [False], 'n_estimators':[3, 10]},
]
grid_search = GridSearchCV(clf, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

# Get the best estimator and predict value
result = grid_search.best_estimator_.predict(X_test)

y_test = [value/10 for value in y_test]
result = result/10

print(f'Result: {result}')
print(f'Real Value: {y_test}')

print(f'RMSE: {mean_squared_error(y_test, result)}')

# Show prediction and ground truth on graph
plt.scatter(X_test, result, color='blue', label='Prediction')
plt.scatter(X_test, y_test, color='red', label='Ground Truth')
plt.show()