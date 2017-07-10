# Imports

from csv import reader
from random import seed
from random import randrange

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

# Spliting dataset into a train set and test set--------

def train_test_split(dataset, split = 0.6):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy
# ------------------------------------------------------

# K-fold cross-validation ------------------------------

def cross_validation_split(dataset, folds = 3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# ------------------------------------------------------

# Main

filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)

seed(1)

# Train and test

train, test = train_test_split(dataset)
print(train)
print(test)

# cross-validation

folds = cross_validation_split(dataset, 4)
print(folds)
