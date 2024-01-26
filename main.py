#!/usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import randomforest as rforest
import tensorflow as tf

# default params to keras.dataset.imdb.load_data
# see: https://keras.io/api/datasets/imdb
start_char = 1
oov_char = 2  # oov = out-of-vocabulary
index_from = 3

# retrieve training sequences
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        start_char=start_char, oov_char=oov_char, index_from=index_from
 )

# retrieve word index and create it's inverse
word_to_index = tf.keras.datasets.imdb.get_word_index()
index_to_word = dict(
    (i + index_from, word) for (word, i) in word_to_index.items()
)

# update inverted word index to include start_char and oov_char
index_to_word[start_char] = '[START]'
index_to_word[oov_char] = '[OOV]'

# decode each review in the dataset
x_train = np.array([' '.join([index_to_word[idx] for idx in text])
                    for text in x_train])
x_test = np.array([' '.join([index_to_word[idx] for idx in text])
                   for text in x_test])

# vectorize each review and create the vocabulary
binary_vectorizer = CountVectorizer(binary=True, min_df=100)
x_train_bin = binary_vectorizer.fit_transform(x_train).toarray()
x_test_bin = binary_vectorizer.fit_transform(x_test).toarray()
vocab = np.array(list(binary_vectorizer.vocabulary_.keys()))

# TODO force constraints on vocabulary words
# 1. contains the m most common,
# 2. ignores the n most common,
# 3. ignores the k rarest.

if __name__ == '__main__':
    sel_alg = int(input('Select an algorithm:\n[1]: Naive Bayes \
            \n[2]: Random Forest\n[3]: Adaboost\nYour selection: '))
    if sel_alg > 3 or sel_alg < 1:
        print('Invalid selection.')
        exit(1)
    elif sel_alg == 1:
        print('Not yet implemented.')
        exit(2)
    elif sel_alg == 2:
        numberOfTrees = int(input('Number of trees (integer value): '))
        print(f'Creating a Random Forest with {numberOfTrees} trees...')
        rforest.RandomForest(numberOfTrees, (x_test, y_test))
    elif sel_alg == 3:
        print('Not yet implemented.')
        exit(2)
