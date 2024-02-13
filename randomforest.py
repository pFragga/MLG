#!/usr/bin/env python

from random import randint
from statistics import mode
import id3
import numpy as np


class RandomForest:
    def __init__(self, num_trees, features):
        self.num_trees = num_trees
        self.trees = []
        self.features = features
        self.predictions = []

    def bootstrap(self, dataset):
        """
        Create a random sample with replacement of the provided `dataset`.
        >>> dataset = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])
        >>> bootstrap(dataset)
        array([[0, 1, 1],
               [0, 1, 1],
               [1, 1, 1]])
        >>> bootstrap(dataset)
        array([[1, 0, 1],
               [1, 0, 1],
               [0, 1, 1]])
        """
        _dataset = []
        for sample in dataset:
            _dataset.append(dataset[randint(0, len(dataset) - 1)])
        return np.array(_dataset)

    def fit(self, x, y):
        """
        ## Bagging - Part 1
        Given a training set `x` with responses `y`, bagging repeatedly
        (`self.num_trees` times) creates a bootstrapped training set and fits
        decision trees to these samples using the `ID3` algorithm.

        `ID3` has been modified so that at each candidate split, only a random
        subset of the provided features is considered.
        """
        for i in range(self.num_trees):
            _x = self.bootstrap(x)
            tree = id3.ID3(self.features)
            tree.fit(_x, y)
            self.trees.append(tree)  # trees ends up containing ID3 objects...

    def aggregate(self, predictions):
        """
        A prediction is just an array of classifications for samples of data.
        E.g. `predictions[0][0] == 0`, means the first prediction, classified
        the first sample as category 0.

        By "aggregating" the `predictions`, we construct a new prediction where
        we classify each sample based on how the majority of the provided
        predictions classified that specific sample (0 or 1).
        >>> predictions = np.array([[0, 1], [0, 0], [1, 1], [0, 1])
        >>> aggregate(predictions)
        array([0, 1])
        """
        prediction = []
        for j in range(len(predictions[0])):
            tmp = []
            for i in range(len(predictions)):
                tmp.append(predictions[i][j])
            prediction.append(mode(tmp))
        return np.array(prediction)

    def predict(self, x):
        """
        ## Bagging - Part 2
        After training, predictions for unseen samples `x` can be made by
        aggregating the predictions from all the individual trees by taking a
        majority vote.
        """
        if len(self.trees) == 0:
            return None

        for tree in self.trees:
            self.predictions.append(tree.predict(x))
        return self.aggregate(self.predictions)


if __name__ == '__main__':
    import main
