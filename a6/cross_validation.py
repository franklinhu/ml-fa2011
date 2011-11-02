# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
import math
import random

class CrossValidation:
    
    def __init__(self, num_folds, data, shuffle=False):
        self.num_folds = num_folds
        self.data = list(data)
        self.items_per_fold = int(math.ceil(float(len(data)) / num_folds))

        if shuffle:
            random.shuffle(self.data)

    def training_examples(self, fold_num):
        start,stop = self._training_index_range(fold_num)
        for i in xrange(start, stop):
            yield self.data[i]

    def validation_examples(self, fold_num):
        start,stop = self._training_index_range(fold_num)
        for i in xrange(len(self.data)):
            if i >= start and i < stop:
                continue
            yield self.data[i]

    def _training_index_range(self, fold_num):
        if fold_num >= self.num_folds:
            raise Exception("Invalid fold number: %d" % fold_num)
        start = fold_num * self.items_per_fold
        stop = min(len(self.data), start + self.items_per_fold)
        return start, stop
