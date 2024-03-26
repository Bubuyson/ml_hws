# Question 2
import numpy as np
import matplotlib.pyplot as plt

# This is a helper function to create a design matrix for polynomial regression see the link below for more details
# https://media.licdn.com/dms/image/D4D22AQHb8Tn79irlXA/feedshare-shrink_800/0/1698072977775?e=2147483647&v=beta&t=d86zVjVUHgaleFSRZDT5UaI_IUqTfQXQCDWIFBHAM8I
def design_matrix(x, degree):
    X = np.ones((len(x), 1))
    for i in range(1, degree + 1):
        X = np.hstack((X, np.power(x, i).reshape(-1, 1)))
    return X

# Closed-form solution to find the coefficients of the polynomial regression
# Formula: (X^T * X)^-1 * X^T * y
def polynomial_regression_closed_form(x, y, degree):
    X = design_matrix(x, degree)
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffs

# Gradient Descent Algorithm
# Formula: theta = theta - learning_rate * gradients where gradient is the derivative of the cost function with respect to the polynomial coefficients
def gradient_descent(X, y, learning_rate, iterations, degree, error_type=''):
    m = len(y)
    theta = 0.5 * np.random.randn(degree + 1, 1)  # random initialization of parameters 

    for _ in range(iterations):
        if error_type == 'mean_squared':
            gradients = 2/m * X.T @ (X @ theta - y)
        elif error_type == 'mean_absolute':
            gradients = 2/m * X.T @ (np.sign(X @ theta - y))
        else:
            print('Error type not recognized')
            return
        
        theta = theta - learning_rate * gradients
    return theta

# Calculate the expected absolute error
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
            print(f'The expected absolute error for the {data_name} data using {method} and degree {degree} is {error:.7f}')
        else:
            print(f'The expected absolute error for the {data_name} data using {method} is {error:.7f}')
    return error

# Plot the data along with the regression line
def plot_data(x, y, coeffs, title, suptitle, degree):
    for i in range(len(x)):
        plt.subplot(1, len(x), i + 1)
        plt.scatter(x[i], y[i], color='blue')
        x_values = np.linspace(min(x[i]) - 0.1, max(x[i]) + 0.1, 100)
        y_values = np.polyval(coeffs[i][::-1], x_values)
        plt.plot(x_values, y_values, color='red')
        plt.title(title[i] + ' Case with Regression Line for Degree ' + str(degree))
        plt.xlabel('x')
        plt.ylabel('y')
    plt.suptitle(suptitle)
    plt.show()

if __name__ == '__main__':

    x_example = [None] * 3
    y_example = [None] * 3

    # Data for the linear case
    x_example[0] = np.array([3.6094667 , 3.27761855, 2.3375142 , 1.30156752, 2.09238962,
            2.2086992 , 2.98037837, 1.50848853, 2.8796192 , 2.18324089,
            1.21758807, 2.95593106, 3.16605344, 1.75270183, 3.66610873,])
    y_example[0] = np.array([3.86684234, 2.98013852, 2.5225595 , 1.33820623, 1.815342  ,
            2.16810137, 3.05943711, 1.59362909, 2.82437503, 2.33595141,
            1.40077359, 2.74390262, 3.41048636, 1.60432259, 3.73007504])

    # Data for the outlier case
    x_example[1] = np.array([4, 1.27352097, 1.43341861, 1.12349433, 1.42553163, 1.54455854,
            1.94893626, 1.60707131, 1.45906907, 1.83648581, 1.62540045,
            1.38588989, 1.66016047, 1.47585759, 1.77348134])
    y_example[1] = np.array([4, 2.51600608, 2.74140655, 1.67086539, 2.78758829, 3.02421673,
            4.25971091, 3.38695732, 2.98598498, 4.10311279, 3.4986229 ,
            2.73863463, 3.57128138, 3.09710574, 3.66806792])

    # Data for the polynomial case
    x_example[2] = np.array([1.        ,
            1.2       , 1.4       , 1.6       , 1.8       , 2.        ,
            2.2       , 2.4       , 2.6       , 2.8       , 3.        ,
            3.2       , 3.4       , 3.6       , 3.8])
    y_example[2] = np.array([1.06085075,
            1.46015222, 1.79504318, 2.10876087, 2.33798963, 2.52933699,
            2.67944026, 2.89255978, 2.88517906, 2.89533251, 3.01370418,
            3.05801204, 2.82661187, 2.77394949, 2.63437779])


    error_type = 'mean_squared'

    # Map of data names to their respective indices
    data_names = {0: 'Linear', 1: 'Outlier', 2: 'Polynomial'}

    degree = 1
    # degree = 4

    # Now set up the gradient descent parameters
    learning_rate = 0.0001
    iterations = 100000

    coeffs_closed_form = [None] * 3
    for i in range(3):
        coeffs_closed_form[i] = polynomial_regression_closed_form(x_example[i], y_example[i], degree)

    X_design = [None] * 3
    theta_gradient_descent = [None] * 3
    for i in range(3):
        X_design[i] = design_matrix(x_example[i], degree)
        theta_gradient_descent[i] = gradient_descent(X_design[i], y_example[i].reshape(-1, 1), learning_rate, iterations, degree, error_type)

    # Calculate expected absolute error for both models
    error_closed_form = [None] * 3
    error_gradient_descent = [None] * 3

    for i in range(3):
        error_closed_form[i] = calculate_expected_absolute_error(X_design[i], y_example[i], coeffs_closed_form[i], True, 'closed_form', data_names[i], degree, error_type)
        error_gradient_descent[i] = calculate_expected_absolute_error(X_design[i], y_example[i].reshape(-1, 1), theta_gradient_descent[i], True, 'gradient_descent', data_names[i], degree, error_type)


    plt.figure(figsize=(18, 6))
    plot_data(x_example, y_example, coeffs_closed_form, data_names, 'Closed Form Solution', degree)

    plt.figure(figsize=(18, 6))
    plot_data(x_example, y_example, theta_gradient_descent, data_names, 'Gradient Descent Solution', degree)

