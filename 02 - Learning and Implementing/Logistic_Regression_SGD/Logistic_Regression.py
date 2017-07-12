# Logistic regression on Diabetes Dataset

# Imports

from random import seed
from random import randrange
from csv import reader
from math import exp

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

# Str to float

def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find minmax values

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

# Split data in k-folds

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

# Calculate accuracy percentage

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100

# Evaluate an algo using cross val

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
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Prediction with coefficients

def predict(row, coefficients):
    pred_y = coefficients[0]
    for i in range(len(row) - 1):
        pred_y += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-pred_y))

# Estimate coefficients

def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            pred_y = predict(row, coef)
            error = row[-1] - pred_y
            coef[0] = coef[0] + l_rate * error * pred_y * (1.0 - pred_y)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * pred_y * (1.0 - pred_y) * row[i]
    return coef

# Logistic Regression

def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        pred_y = predict(row, coef)
        pred_y = round(pred_y)
        predictions.append(pred_y)
    return predictions

# Main

seed(1)

filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_to_float(dataset, i)

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_folds = 5
l_rate = 0.1
n_epochs = 100

scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epochs)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
