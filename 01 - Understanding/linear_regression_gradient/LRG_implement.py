# Linear Regression with Gradient Descent implementation

# Gradient descent

# open + read

with open('x_input.in') as fx:
    x = fx.readlines()

with open('y_output.in') as fy:
    y = fy.readlines()

fx.close()
fy.close()

# numer of epochs and learning rate

epochs = 4
alpha = 0.01

# preset B0 and B1(bias and weight)

B0 = [0.0] * (epochs * len(x))
B1 = [0.0] * (epochs * len(x))

# itterations

predicted_y = []

for i in range(0, epochs * len(x)):
    predicted_y.append(B0 + B1 * int(y[i]))
    error = float(predicted_y[i]) - int(y[i])
    B0[i+1] = B0[i] - alpha * error
    B1[i+1] = B1[i] - alpha * error * int(x[i])

print(B0)
print(B1)

# Apply Simple Linear Regression with B0 and B1 discovered earlier
# And that's it
