# Diabetes Regression Models

## Purpose

The purpose of this project is to build and compare different machine learning regression models using the built-in diabetes dataset from Scikit-Learn. The models are trained to predict diabetes disease progression based on patient medical features.

## Implementation

The program loads the diabetes dataset, divides it into sets for training and testing, and trains three models for regression.
* Random Forest Regressor
* Linear Regression
* K-Nearest Neighbors Regressor

The training data is used to train each model, and then the model is used to guess values for the test data. Three metrics are used to measure how well each model works:
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

The project also looks at how changing the model parameters affects how well it works. The number of trees (estimators) is different for Random Forest. The number of neighbors is changed for K-Nearest Neighbors.

## Classes and Methods

This project does not implement custom classes. Instead, it uses machine learning classes provided by Scikit-Learn.

### Models Used

* "RandomForestRegressor" is a model that uses a group of decision trees to make predictions.
* "LinearRegression" is a simple regression model that fits a linear equation to the data.
* `KNeighborsRegressor` is a model that uses the average of nearby data points to make predictions.

### Methods Used

* `fit()` – Uses the training dataset to teach the model.
* `predict()` - Uses the trained model to make predictions.
* `train_test_split()` - Splits the dataset into two parts: one for training and one for testing.

### Evaluation Methods

* `mean_absolute_error()` – Measures the average prediction error.
* `mean_squared_error()` – Measures the squared prediction error.
* `r2_score()` – Measures how well the model explains variance in the data.

## Limitations
This project uses a small dataset and the default settings for most models. Because of this, the models might not make the best predictions possible. More tuning and bigger datasets might make performance better.
