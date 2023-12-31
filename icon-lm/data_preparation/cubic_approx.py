from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Define the function sin(x) - cos(x)
def original_function(x):
    return np.sin(x) - np.cos(x)

# Define the cubic function to fit
def cubic_function(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def fix_x_in_range(xmin, xmax):
    # Generate x values in the range [-1, 1]
    x_values = np.linspace(xmin, xmax, 100)
    # Compute y values for these x values using the original function
    y_values = original_function(x_values)
    # Fit the cubic function to the original function
    parameters, _ = curve_fit(cubic_function, x_values, y_values)
    return parameters

# Extract the parameters for the cubic approximation
print(fix_x_in_range(-1, 1))

# Plotting
plt.figure(figsize=(20, 5))
for xmin, xmax, i in zip([-1, -2, -3], [1, 2, 3], range(1, 4)):
    plt.subplot(1, 3, i)
    x_plot = np.linspace(xmin, xmax, 100)
    plt.plot(x_plot, original_function(x_plot), label='y = sin(x) - cos(x)', color='black')
    params = fix_x_in_range(xmin, xmax)
    plt.plot(x_plot, cubic_function(x_plot, *params), label='{:.3f}x^3 + {:.3f}x^2 + {:.3f}x + {:.3f}'.format(*params), color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Approximation in the range [{}, {}]'.format(xmin, xmax))
    plt.legend()
    plt.grid(True)
plt.savefig('cubic_approx.png')