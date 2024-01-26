#!/usr/bin/env python

import id3
import numpy as np


class RandomForest:
    def __init__(self, numberOfTrees: int, features: iter):
        self.numberOfTrees = numberOfTrees
        self.trees = []
        self.features = features

    def bootstrap(self, x, y):
        """
        First step of "bagging" procedure.
        """
        pass

    def aggregate(self, x, y):
        """
        Second step of "bagging" procedure.
        """
        pass

    def train(self, x, y):
        """
        Here x and y are the training data.
        """
        pass

    def fit(self, x, y):
        """
        Here x and y are the test data.
        """
        pass
