# Question 2
import numpy as np
import matplotlib.pyplot as plt

# This is a helper function to create a design matrix for polynomial regression
def design_matrix(x, degree):
    X = np.ones((len(x), 1))
    for i in range(1, degree + 1):
        X = np.hstack((X, np.power(x, i).reshape(-1, 1)))
    return X

# Closed-form solution to find the coefficients of the polynomial regression
def polynomial_regression_closed_form(x, y, degree):
    X = design_matrix(x, degree)
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffs

# Gradient Descent Algorithm
def gradient_descent(X, y, learning_rate, iterations, degree):
    m = len(y)
    theta = np.random.randn(degree + 1, 1)  # random initialization of parameters 

    for iteration in range(iterations):
        gradients = 2/m * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradients

    # return to python list
    # theta = theta.tolist()
    # theta = np.array(theta)
    return theta

def calculate_expected_absolute_error(X, y, theta, verbose=False, method='closed_form', data_name='', degree=None, error_type=''):
    y_pred = X @ theta
    if error_type == 'mean_squared':
        error = np.square(y_pred - y).mean()
    elif error_type == 'mean_absolute':
        error = np.abs(y_pred - y).mean()
    else:
        print('Error type not recognized')
        return
    
    if verbose:
        if degree is not None:
            print(f'The expected absolute error for the {data_name} data using {method} and degree {degree} is {error:.4f}')
        else:
            print(f'The expected absolute error for the {data_name} data using {method} is {error:.4f}')
    return error

def plot_data(x, y, coeffs, title, suptitle, degree):
    for i in range(len(x)):
        plt.subplot(1, len(x), i + 1)
        plt.scatter(x[i], y[i], color='blue')
        x_values = np.linspace(-2, 2, 1000)
        y_values = np.polyval(coeffs[i][::-1], x_values)
        plt.plot(x_values, y_values, color='red')
        plt.title(title[i] + ' Case with Regression Line for Degree ' + str(degree))
        plt.xlabel('x')
        plt.ylabel('y')
    plt.suptitle(suptitle)
    plt.show()

x_example = [None] * 3
y_example = [None] * 3

# Data for the 'outlier' case
x_example[0] = np.array([-2, -1.8, -1, -0.5, 0, 0.5, 1, 1.5, 1.7, 2])
y_example[0] = np.array([2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, 8, 9])

# Data for the 'Poly' (polynomial) case
x_example[1] = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
y_example[1] = np.array([4, 2, 0.5, 0.2, 0, 0.2, 0.5, 2, 4])

# Data for the 'Linear' case
x_example[2] = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
y_example[2] = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

# Map of data names to their respective indices
data_names = {0: 'Outlier', 1: 'Poly', 2: 'Linear'}

degree = 1

coeffs_closed_form = [None] * 3
for i in range(3):
    coeffs_closed_form[i] = polynomial_regression_closed_form(x_example[i], y_example[i], degree)

# Now set up the gradient descent parameters
learning_rate = 0.01
iterations = 1000

# Prepare the design matrix for gradient descent
X_design = [None] * 3
theta_gradient_descent = [None] * 3
for i in range(3):
    X_design[i] = design_matrix(x_example[i], degree)
    theta_gradient_descent[i] = gradient_descent(X_design[i], y_example[i].reshape(-1, 1), learning_rate, iterations, degree)

# Calculate expected absolute error for both models
error_closed_form = [None] * 3
error_gradient_descent = [None] * 3

# print('Expected Absolute Errors:' + ' for degree ' + str(degree))
for i in range(3):
    error_closed_form[i] = calculate_expected_absolute_error(X_design[i], y_example[i], coeffs_closed_form[i], True, 'closed_form', data_names[i], degree, 'mean_squared')
    error_gradient_descent[i] = calculate_expected_absolute_error(X_design[i], y_example[i].reshape(-1, 1), theta_gradient_descent[i], True, 'gradient_descent', data_names[i], degree, 'mean_squared')

# Plot for the 'Outlier' case
plt.figure(figsize=(18, 6))
plot_data(x_example, y_example, coeffs_closed_form, data_names, 'Closed Form Solution', degree)

# Plot the gradient descent solution along with the data points in a 3 x 1 subplot
plt.figure(figsize=(18, 6))
plot_data(x_example, y_example, theta_gradient_descent, data_names, 'Gradient Descent Solution', degree)




