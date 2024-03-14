# Question 5

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def laplace_distribution(x):
    return 0.5 * np.exp(-np.abs(x))

x_values = np.linspace(-5, 5, 1000)
y_values = laplace_distribution(x_values)
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Laplace Distribution')
plt.grid(True)
plt.show()

prob_x_gt_2 = quad(laplace_distribution, 2, np.inf)
print('The probability that x > 2 is', prob_x_gt_2[0])

