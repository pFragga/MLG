#!/usr/bin/env python

import id3
import numpy as np


class RandomForest:
    numberOfTrees = 2  # default value, change to your liking

    def __init__(self, numberOfTrees: int, features: iter):
        self.numberOfTrees = numberOfTrees
        self.features = features

    def permutate(self, data: iter, k: int):
        """
        Create a random permutation of the given dataset. Return the new
        dataset excluding the k first elements.
        """
        new_data = np.random.permutation(data)
        return new_data[k:]

    def fit(self, x, y):
        """
        Create all trees in the forest
        """
        for i in range(self.numberOfTrees):
            id3.ID3.fit(x, y)
