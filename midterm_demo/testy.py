import numpy as np
import matplotlib.pyplot as plt
import math

#define histogram estimator function.
def histogram_estimator(x, h, training_data):
    #calculate the distance between the point x and each training data point.
    distances = np.linalg.norm(x - training_data)
    #calculate the kernel weight for each training data point.
    kernel_weights = kernel(distances, h)
    #count the number of training data points within a distance h of x.
    count = np.sum(kernel_weights)
    #normalize the count by the kernel bandwidth.
    density_estimate = count / (h * len(training_data))
    return density_estimate

##define the kernel function:
def kernel(distances, h):
    return np.exp(-(distances**2) / (2 * h**2))

##load the training data.
training_data = np.array([0.2, 1.3, 1.8, 3.2, 5.0, 6.2, 6.8, 8.5, 9.2, 9.9])

##Calculate the density estimate at each point in the range [0, 25]
x_values = np.arange(0, 10, 0.1)
y_values = [histogram_estimator(x, 5, training_data) for x in x_values]

#plot histogram estimator
plt.plot(x_values, y_values, linewidth=2)
plt.xlabel("x")
plt.ylabel("Histogram estimator")
plt.title("Histogram estimator for training data using h = 5")
plt.show()