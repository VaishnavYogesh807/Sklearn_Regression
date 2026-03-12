# Import required modules from sklearn
from sklearn import datasets, model_selection, ensemble, linear_model, neighbors, metrics

# Load the built-in diabetes dataset
diab = datasets.load_diabetes()

# Separate the dataset into features (x) and target variable (y)
x = diab.data
y = diab.target

# Split the dataset into training data (80%) and testing data (20%)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# Print the shape of each dataset to verify the split
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Create and train a Random Forest regression model
forest_model = ensemble.RandomForestRegressor()
forest_model.fit(x_train, y_train)

# Use the trained model to predict values for the test set
forest_pred = forest_model.predict(x_test)

# Create and train a Linear Regression model
lin_model = linear_model.LinearRegression()
lin_model.fit(x_train, y_train)

# Predict values using the Linear Regression model
linear_pred = lin_model.predict(x_test)

# Create and train a K-Nearest Neighbors regression model
neighbor_model = neighbors.KNeighborsRegressor()
neighbor_model.fit(x_train, y_train)

# Predict values using the KNN model
neighbor_pred = neighbor_model.predict(x_test)

# Print Mean Absolute Error (MAE) for each model
print("Random Forest MAE:", metrics.mean_absolute_error(y_test, forest_pred))
print("Linear Regression MAE:", metrics.mean_absolute_error(y_test, linear_pred))
print("K-Nearest Neighbors MAE:", metrics.mean_absolute_error(y_test, neighbor_pred))
print()

# Print Mean Squared Error (MSE) for each model
print("Random Forest MSE:", metrics.mean_squared_error(y_test, forest_pred))
print("Linear Regression MSE:", metrics.mean_squared_error(y_test, linear_pred))
print("K-Nearest Neighbors MSE:", metrics.mean_squared_error(y_test, neighbor_pred))
print()

# Print R^2 score for each model
print("Random Forest R^2:", metrics.r2_score(y_test, forest_pred))
print("Linear Regression R^2:", metrics.r2_score(y_test, linear_pred))
print("K-Nearest Neighbors R^2:", metrics.r2_score(y_test, neighbor_pred))
print()

#Analysis:
# The Linear Regression model performed the best overall. 
# While the K-Nearest Neighbors model had the lowest MAE (42.77), it had the highest MSE (3019.08) and a lowest R² score (0.43) compared to the Linear Regression model.
# It still produced a low MAE (42.79) and a low MSE (2900.19), meaning its predictions were closest to the actual values on average. 
# It also achieved the highest R² score (0.45), indicating it explained the largest portion of variance in the data.

# List of values to test for model parameters
estimator_list = [10, 50, 100, 200]

# Test Random Forest performance with different numbers of trees
for n in estimator_list:
    forest_model = ensemble.RandomForestRegressor(n_estimators=n)
    forest_model.fit(x_train, y_train)
    forest_pred = forest_model.predict(x_test)
    
    # Print evaluation metrics for each configuration
    print(f"Random Forest with {n} estimators - MAE: {metrics.mean_absolute_error(y_test, forest_pred):.2f}, MSE: {metrics.mean_squared_error(y_test, forest_pred):.2f}, R^2: {metrics.r2_score(y_test, forest_pred):.2f}")

print()

# Test KNN performance with different numbers of neighbors
for n in estimator_list:
    neighbor_model = neighbors.KNeighborsRegressor(n_neighbors=n)
    neighbor_model.fit(x_train, y_train)
    neighbor_pred = neighbor_model.predict(x_test)
    
    # Print evaluation metrics for each configuration
    print(f"K-Nearest Neighbors with {n} neighbors - MAE: {metrics.mean_absolute_error(y_test, neighbor_pred):.2f}, MSE: {metrics.mean_squared_error(y_test, neighbor_pred):.2f}, R^2: {metrics.r2_score(y_test, neighbor_pred):.2f}")

#Analysis:
# For the Random Forest model, when calculating the MAE, MSE and R^2, the performance improved as the number of estimators increased from 10 to 50, but then plateaued with 100 and 200 estimators. 
# For the K-Nearest Neighbors model, the performance was best with 10 neighbors for MAE, but not for MSE or R^2. The model began to drop as the number of neighbors increased, with the worst performance at 200 neighbors.
# For the Linear Regression model, since it does not have a parameter like n_estimators or n_neighbors, we cannot perform a similar analysis. However, it consistently outperformed the K-Nearest Neighbors model across all metrics, and had comparable performance to the Random Forest model with 50 or more estimators.