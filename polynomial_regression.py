# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# position_salaries.csv contains two feature (Non-Linear Data)
print('-------------------------------------------------')
print('Seprating features and dependent variable . . . ')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# As dataset's rows are less, Splitting won't be necessary

# Spliting
# from sklearn.model_selection import train_test_split
# print('-------------------------------------------------')
# print('splitting . . . ')
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Displaying
print('-------------------------------------------------')
print('Length of X', len(X))
print(X)
print('-------------------------------------------------')
print('Length of y_train', len(y))
print(y)

# Machine learns
from sklearn.linear_model import LinearRegression
print('-------------------------------------------------')
print('Machine is learning . . .')
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
# As Polynomial Linear Regeression is special case of Multiple Linear Regression
# fitting x,x2,x3 . . .
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures() # best value of degree according to data set. Tested
X_fitted = poly_regressor.fit_transform(X)
final_regressor = LinearRegression()

# Learning and Predicting
print('-------------------------------------------------')
print('Predicting . . .')
final_regressor.fit(X_fitted, y)

print('-------------------------------------------------')
print('Visualising the Result')
# Prediction of HardCoded Value
print("Prediction of hard Coded Value:")
print("  With Linear Regressor: ",linear_regressor.predict([[6.5]]))
print("  With Linear Regressor: ",final_regressor.predict(poly_regressor.fit_transform([[6.5]])))

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, final_regressor.predict(poly_regressor.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, final_regressor.predict(poly_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()