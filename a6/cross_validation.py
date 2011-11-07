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

    def validation_examples(self, fold_num):
        """Returns a generator that gives the validation examples for a
        particular fold

        fold_num -- Fold number in range of [0, self.num_folds-1]
        """
        start,stop = self._validation_index_range(fold_num)
        for i in xrange(start, stop):
            yield self.data[i]

    def training_examples(self, fold_num):
        """Returns a generator that gives the training examples for a 
        particular fold
        """
        start,stop = self._validation_index_range(fold_num)
        for i in xrange(len(self.data)):
            if i >= start and i < stop:
                continue
            yield self.data[i]

    def num_training_examples(self, fold_num):
        """Returns the number of training examples in a particular fold
        """
        return len(self.data) - self.num_validation_examples(fold_num)

    def num_validation_examples(self, fold_num):
        """Returns the number of validation examples in a particular fold
        """
        start,stop = self._validation_index_range(fold_num)
        return stop - start

    def _validation_index_range(self, fold_num):
        if fold_num >= self.num_folds:
            raise Exception("Invalid fold number: %d" % fold_num)
        start = fold_num * self.items_per_fold
        stop = min(len(self.data), start + self.items_per_fold)
        return start, stop

