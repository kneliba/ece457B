import random
import matplotlib.pyplot as plt
import numpy as np
import math


def perceptron(dimension, tolerance, factor, inputs, target):
    weights = [random.uniform(-1, 1)] * dimension
    print(len(inputs[0][0]))
    updated = True
    while updated == True:
        updated = False

        for i in range(len(inputs)):
            input_set = inputs[i]

            for j in range(len(input_set)):
                sum = 0
                for k in range(dimension):
                    sum += input_set[j][k]*weights[k]
                if sum > 0:
                    o = 1
                else:
                    o = 0
                if o != target[i]:
                    for l in range(dimension):
                        change = factor*(target[i]-o)*input_set[j][l]
                        # print(change)

                        if abs(change) > tolerance:
                            updated = True

                        weights[l] += change
    print(weights)
    return weights


def adaline(dimension, tolerance, factor, inputs, target):
    weights = [random.uniform(-1, 1)] * dimension
    print(len(inputs[0][0]))
    updated = True
    while updated == True:
        updated = False

        for i in range(len(inputs)):
            input_set = inputs[i]

            for j in range(len(input_set)):
                sum = 0
                for k in range(dimension):
                    sum += input_set[j][k]*weights[k]
                if sum > 0:
                    o = 1
                else:
                    o = 0
                if o != target[i]:
                    for l in range(dimension):
                        s = 1/(1+math.exp(-sum))
                        change = factor*(target[i]-s) * \
                            s**2 * math.exp(-sum) * input_set[j][l]
                        # print(change)

                        if abs(change) > tolerance:
                            updated = True

                        weights[l] += change
    return weights


input = [[[-1, 0.8, 0.7, 1.2], [-1, -0.8, -0.7, 0.2], [-1, -0.5, 0.3, -0.2], [-1, -2.8, -0.1, -2]],
         [[-1, 1.2, -1.7, 2.2], [-1, -0.8, -2, 0.5], [-1, -0.5, -2.7, -1.2], [-1, 2.8, -1.4, 2.1]]]

target = [0, 1]

dimension = 4
tolerance = 0.01
factor = 0.6

weights = perceptron(dimension, tolerance, factor, input, target)
print(weights)


ax = plt.axes(projection='3d')

x = np.outer(np.linspace(-3, 3, 100), np.ones(100))
y = x.copy().T
z = (weights[1]*x + weights[2]*y - weights[0]) / - weights[3]

ax.plot_surface(x, y, z, color="blue")

for i in range(len(input)):
    for j in range(len(input[i])):
        if i == 0:
            ax.scatter(input[i][j][1], input[i][j][2],
                       input[i][j][3], color='green')
        else:
            ax.scatter(input[i][j][1], input[i][j][2],
                       input[i][j][3], color='red')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(
    f'z = ({weights[1]}*x + {weights[2]}*y - {weights[0]}) / ( - {weights[3]})')
ax.scatter(-1.4, -1.5, 2, color='purple')
plt.show()
