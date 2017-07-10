# Imports

from csv import reader
from math import sqrt

# Load file

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string columns to float

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Normalize Data --------------------------------------------------------------

# Find the max and min for each column

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0 - 1

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
# -----------------------------------------------------------------------------

# Standardize Data ------------------------------------------------------------

# Calculate column means

def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means

# Calculate column starndard deviations

def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs

# Standardize data

def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
# -----------------------------------------------------------------------------

# Main

filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

print(dataset[0])

# normalize -------------------------------
minmax = dataset_minmax(dataset)

normalize_dataset(dataset, minmax)

print(dataset[0])
# -----------------------------------------

# standardize -----------------------------
means = column_means(dataset)

stdevs = column_stdevs(dataset, means)

standardize_dataset(dataset, means, stdevs)

print(dataset[0])
# -----------------------------------------
