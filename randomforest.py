#!/usr/bin/env python

from random import randint
from statistics import mean
import id3
import numpy as np


class RandomForest:
    def __init__(self, numberOfTrees, features):
        self.numberOfTrees = numberOfTrees
        self.trees = []
        self.features = features
        self.results = []

    def bootstrap(self, dataset):
        """
        First step of "bagging" procedure.
        """
        _dataset = []
        for sample in dataset:
            _dataset.append(dataset[randint(0, len(dataset) - 1)])
        return np.array(_dataset)

    def aggregate(self, results):
        """
        Second step of "bagging" procedure.
        """
        return mean(results)

    def train(self, x, y):
        """
        Here x and y are the training data.
        """
        _x = self.bootstrap(x)

        # TODO randomly select features at each step of each algorithm
        for i in range(self.numberOfTrees):
            tree = id3.ID3(self.features)
            tree.fit(_x, y)
            self.trees.append(tree)  # trees ends up containing ID3 objects :/

    def fit(self, x, y):
        """
        Here x and y are the test data.
        """
        for tree in self.trees:
            self.results.append(tree.predict(x))
        return np.array(self.results)
