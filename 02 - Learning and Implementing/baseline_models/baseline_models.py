# Making Random Predictions

from random import seed
from random import randrange

# Generate random predictions

def random_algorithm(train, test):
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = list()
    for row in test:
        index = randrange(len(unique))
        predicted.append(unique[index])
    return predicted

#seed(1)
#train = [[0], [1], [0], [1], [0], [1]]
#test = [[None], [None], [None], [None]]
#predictions = random_algorithm(train, test)
#print(predictions)


# Zero Rule Classification Predictions

# zero rule algorithm for classification

def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

#seed(1)
#train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
#test = [[None], [None], [None], [None]]
#predictions = zero_rule_algorithm_classification(train, test)
#print(predictions)


# Zero Rule Regression Predictions

# zero rule algorithm for regression

def zero_rule_algorithm_regression(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(output_values) / float(len(output_values))
    predicted = [prediction for i in range(len(test))]
    return predicted

seed(1)
train = [[10], [15], [12], [15], [18], [20]]
test = [[None], [None], [None], [None]]
predictions = zero_rule_algorithm_regression(train, test)
print(predictions)
