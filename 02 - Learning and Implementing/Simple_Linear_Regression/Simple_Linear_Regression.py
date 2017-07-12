# Simple Linear Regression on the Swedish Insurance Dataset

from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Loading file

def load_csv(filenname):
    dataset = list()
    with open(filenname, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Conversion

def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Test & Train data split

def tt_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# RMSE

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate the algo

def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = tt_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse

# Mean value

def mean(values):
    return sum(values) / float(len(values))

# Variance value

def variance(values, mean):
    return sum([(x-mean) ** 2 for x in values])

# Covariance value

def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i]- mean_y)
    return covar

# Bias and weights ( B0 && B1)

def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

# Simple Linear Regression algo

def SLR(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        pred_y = b0 + b1 * row[0]
        predictions.append(pred_y)
    return predictions

# Main --------------------------------------------------------------

seed(1)

filename = 'insurance.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_to_float(dataset, i)

split = 0.6
rmse = evaluate_algorithm(dataset, SLR, split)

print('RMSE: %.3f' % (rmse))
