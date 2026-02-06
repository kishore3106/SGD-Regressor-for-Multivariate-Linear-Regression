# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Load and Prepare the Dataset - Import the dataset and extract multiple independent variables X = [x1, x2, ..., xn] and dependent variable Y.

Step 2: Split the Data - Divide the dataset into training and testing sets (e.g., 80-20 split) using train_test_split.

Step 3: Feature Scaling - Normalize or standardize the features using StandardScaler to ensure all features are on the same scale.

Step 4: Initialize SGD Regressor - Create SGD Regressor model with parameters: learning rate η, max iterations, and loss function (squared loss).

Step 5: Train the Model Using Stochastic Gradient Descent - Fit the model on training data where parameters θ are updated as θ = θ - η∇J(θ) for each sample.

Step 6: Predict and Evaluate - Make predictions on test data using Ŷ = θ0 + θ1x1 + θ2x2 + ... + θnxn and evaluate using metrics (MSE, R² score).

## Program:
```
/*
Developed by: KISHORE B
RegisterNumber: 212224100032

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
x= data.data[:,:3]
y=np.column_stack((data.target,data.data[:,6]))
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state =42)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.fit_transform(y_test)
sgd = SGDRegressor(max_iter=1000, tol = 1e-3)
multi_output_sgd= MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, y_train)
y_pred =multi_output_sgd.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
print(y_pred)
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
   
*/
```

## Output:
<img width="553" height="333" alt="image" src="https://github.com/user-attachments/assets/344af12a-1018-44cb-a5c6-fc670d2caa71" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
