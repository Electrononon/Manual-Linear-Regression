# Linear Regression Program w/ Gradient Descent
import numpy as np
import random


# Create a random set of data for the model
# with a random slope (m) and y-intercept (b)
def getData():
    x = []
    y = []
    N = 20  # Number of Data Points
    m = random.uniform(-2, 2)
    b = random.uniform(0, 30)
    for _ in range(N):
        x_i = random.uniform(0, 20)
        y_i = m * x_i + b - random.uniform(-1.5, 1.5)
        x.append(x_i)
        y.append(y_i)
    return [x, y], m, b


# Cost Function
def C(x, y, m, b):
    # Cost function is defined as the square of the distance between m(x_i) + b and y_i
    cost = 0
    for i in range(len(x)):
        x_i = x[i]
        y_i = y[i]
        cost += (m*x_i+b - y_i)**2
    return cost


# Find the gradient for the weight
def dCdm(x, y, m, b):
    deriv = 0
    for i in range(len(x)):
        x_i = x[i]
        y_i = y[i]
        deriv += (2*m*x_i**2 + 2*b*x_i - 2*x_i*y_i)
    return deriv


# Find the gradient for the bias
def dCdb(x, y, m, b):
    deriv = 0
    for i in range(len(x)):
        x_i = x[i]
        y_i = y[i]
        deriv += (2*b + 2*m*x_i - 2*y_i)
    return deriv


# Get the initial data
line, orig_m, orig_b = getData()
x = np.array(line[0], dtype="float32")
y = np.array(line[1], dtype="float32")

# Initial Parameters
m = random.uniform(0, 5)  # Start with random values for m and b
b = random.uniform(0, 5)
N = 6000  # Number of iterations
learningRate = 0.0001

# Use gradient descent to minimize the cost function and update the weight and bias
for i in range(N):
    print("Iteration #" + str(i + 1) + ": ")

    # Get the cost
    cost = C(x, y, m, b)

    # Get gradients
    dm = dCdm(x, y, m, b)
    db = dCdb(x, y, m, b)

    # Update weight and bias
    m -= learningRate*dm
    b -= learningRate*db

    print(f"Cost: {cost}")
    print(f"dCdm: {dm} | dCdb: {db}")
    print(f"m: {m} | b: {b}")
    print()

# Compare the original equation to the model's equation
print(f"\nOriginal Line Equation: y = {orig_m}x + {orig_b}")
print(f"Model's Line Equation: y = {m}x + {b}")
