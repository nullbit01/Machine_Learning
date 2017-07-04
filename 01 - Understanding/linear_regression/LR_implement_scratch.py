# Linear Regression implementation

# imports

from math import sqrt

# open + read

with open('x_input.in') as fx:
    x = fx.readlines()

with open('y_output.in') as fy:
    y = fy.readlines()

fx.close()
fy.close()

# mean(x) && mean(y)

mean_x = int(sum(x))/len(x)
mean_y = int(sum(y))/len(y)

# x - mean(x) && y - mean(y)

x_min_mean = x - mean_x
y_min_mean = y - mean_y

# Multiplication

Multiplication = x_min_mean * y_min_mean

# Sum of multiplications(need for B1)

Sum1 = sum(Multiplication)

# Sum of squared x-mean(x)(need for B1)

Sum2 = sum(x_min_mean**2)

# B1(coefficient) && B0(bias)

B1 = Sum1 / Sum2
B0 = mean_y - (B1 * mean_x)

# predicted y

predict_y = B0 + B1 * x

# squared error

squared_error = (predict_y - y)**2
sum_squared_error = sum(squared_error)

# Root Mean Squared Error

RMSE = sqrt(sum_squared_error/sum(x))

print(RMSE)
