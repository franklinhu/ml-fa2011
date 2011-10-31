import math
import random
import sys

import matplotlib.pyplot as plt

import logistic_regression_model

NUM_ARGS = 1
THRESHOLD = 0.0001

def get_validation_error(model, validation):
    errors = []
    for cls,ex in validation:
        if cls == -1:
            cls = 0
        errors.append(abs(model.predict(ex) - cls))
    return sum(errors)

if __name__ == "__main__":
    if len(sys.argv) < NUM_ARGS + 1:
        print "Usage: train.py data_file"
        sys.exit(1)

    data_file = sys.argv[1]
    handle = open(data_file, 'r')
    header = handle.readline().rstrip("\n").split(" ")

    data = []
    for line in handle:
        line = line.rstrip("\n").split(" ")
        cls = -1 if line[0] == '0' else 1
        data.append([cls, [math.log(int(x)) if int(x) > 0 else 0 for x in line[1:]]])
    handle.close()

    num_examples = len(data)
    random.shuffle(data)

    partition = num_examples / 10
    validation = data[:partition]
    training = data[partition:]

    model = logistic_regression_model.LogisticRegressionModel(len(header)-1)

    min_error = float("inf")
    count = 0
    errors = []
    while True:
        cls,ex = random.choice(training)
        model.update_weights(ex, cls)

        error = get_validation_error(model, validation)
        errors.append(error)
        print "%d Error: %f %f" % (count, error, min_error)
        if error <= min_error: 
            if (min_error - error) < THRESHOLD:
                break
            min_error = error

        count += 1
    #plt.plot(range(len(errors)), errors)
    #plt.show()
    print count
