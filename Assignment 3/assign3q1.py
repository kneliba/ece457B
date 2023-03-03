import numpy as np

x_training = [-6, -5, -2, 0, 1, 3, 5]


def orginal_function(x):
    return - np.square(np.abs(x))*np.sin((np.pi/2)*x)


y_training = [orginal_function(x_training) for _ in x_training]
print(y_training)
