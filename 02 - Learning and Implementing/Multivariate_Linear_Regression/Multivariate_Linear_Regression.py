# Linear Regression with Stochastic Gradient Descent for Wine Quality

# Imports

from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load File

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string to float

def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find min and max on each column

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_val = [row[i] for row in dataset]
        val_min = min(col_val)
        val_max = max(col_val)
        minmax.append([val_min, val_max])
    return minmax

# Rescale data to 0-1

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split data in k folds

def cross_validation(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Root Mean Squread Error

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate the algorithm with k-fold

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores

# Predictic with coefficients

def predict(row, coefficients):
    pred_y = coefficients[0]
    for i in range(len(row) - 1):
        pred_y += coefficients[i+1] * row[i]
    return pred_y

# Estimatig coefficients

def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            pred_y = predict(row, coef)
            error = pred_y - row[-1]
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
    return coef

# Linear Regression with SGD

def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        pred_y = predict(row, coef)
        predictions.append(pred_y)
    return predictions

# Main

seed(1)

# Load

filename = 'winequality-white.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_to_float(dataset, i)

# normalize

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# evaluate algorithm

n_folds = 5
l_rate = 0.01
n_epoch = 50
scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores) / float(len(scores))))
