import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Simulated data to approximate the cases shown in the image
# Data for the 'Outlier' case
x_outlier = np.array([-2, -1.8, -1, -0.5, 0, 0.5, 1, 1.5, 1.7, 2])
y_outlier = np.array([2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, 8, 9])

# Data for the 'Poly' (polynomial) case
x_poly = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
y_poly = np.array([4, 2, 0.5, 0.2, 0, 0.2, 0.5, 2, 4])

# Data for the 'Linear' case
x_linear = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
y_linear = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

def compute_least_squares_closed_form(x, y):
    # Add a column of ones to x to account for the bias term (intercept)
    X = np.vstack((np.ones(len(x)), x)).T
    # Calculate coefficients using the normal equation
    # coeffs = (X^T X)^-1 X^T y
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    # Extract the intercept and slope
    intercept, slope = coeffs
    # Calculate the predicted values
    y_pred = X @ coeffs
    return (intercept, slope), y_pred

# Compute least-squares regression line for each case
# coeffs_outlier, y_pred_outlier = compute_least_squares(x_outlier, y_outlier)
# coeffs_poly, y_pred_poly = compute_least_squares(x_poly, y_poly)
# coeffs_linear, y_pred_linear = compute_least_squares(x_linear, y_linear)

coeffs_outlier, y_pred_outlier = compute_least_squares_closed_form(x_outlier, y_outlier)
coeffs_poly, y_pred_poly = compute_least_squares_closed_form(x_poly, y_poly)
coeffs_linear, y_pred_linear = compute_least_squares_closed_form(x_linear, y_linear)

# Plotting the regression lines with the data points
plt.figure(figsize=(18, 6))

# Plot for the 'Outlier' case
plt.subplot(1, 3, 1)
plt.scatter(x_outlier, y_outlier, color='blue')
plt.plot(x_outlier, y_pred_outlier, color='red')
plt.title('Outlier Case with Regression Line')
plt.xlabel('x')
plt.ylabel('y')

# Plot for the 'Poly' case
plt.subplot(1, 3, 2)
plt.scatter(x_poly, y_poly, color='blue')
plt.plot(x_poly, y_pred_poly, color='red')
plt.title('Polynomial Case with Regression Line')
plt.xlabel('x')
plt.ylabel('y')

# Plot for the 'Linear' case
plt.subplot(1, 3, 3)
plt.scatter(x_linear, y_linear, color='blue')
plt.plot(x_linear, y_pred_linear, color='red')
plt.title('Linear Case with Regression Line')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()