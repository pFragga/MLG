#!/usr/bin/env python

import sys

if len(sys.argv[1:]) != 3:
    print('Usage:\n\t[python] ./main.py <m> <n> <k>\nwhere m, n and k are '
          'vocabulary parameters.\n(Hint: try m=1000, n=1000 and k=100)',
          file=sys.stderr)
    exit(-1)

_m = int(sys.argv[1])
_n = int(sys.argv[2])
_k = int(sys.argv[3])

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
    start_char=start_char,
    oov_char=oov_char,
    index_from=index_from,
    num_words=5000  # this only speeds up debugging
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

# vectorize each review using CountVectorizer
# max-df: ignore terms with higher document frequency
# min-df: ignore terms with lower document frequency
binary_vectorizer = CountVectorizer(binary=True,
                                    max_features=_m, max_df=_n, min_df=_k)
x_train_bin = binary_vectorizer.fit_transform(x_train).toarray()
x_test_bin = binary_vectorizer.fit_transform(x_test).toarray()

# CountVectorizer produces a dictionary which maps terms to feature indeces
vocab = np.array(list(binary_vectorizer.vocabulary_.keys()))
print('Printing vocabulary...', vocab, '\n', len(vocab))


def main():
    sel_alg = int(input('Select an algorithm:\n[1]: Naive Bayes \
            \n[2]: Random Forest\n[3]: Adaboost\nYour selection: '))
    if sel_alg > 3 or sel_alg < 1:
        print('Invalid selection.')
        exit(1)
    elif sel_alg == 1:
        print('Not yet implemented.')
        # exit(2)
    elif sel_alg == 2:
        numberOfTrees = int(input('Number of trees (integer value): '))
        print(f'Creating a Random Forest with {numberOfTrees} tree(s)...')
        forest = rforest.RandomForest(numberOfTrees, vocab)
        forest.train(x_train_bin, y_train)
        # forest.fit(x_test_bin, y_test)
    elif sel_alg == 3:
        print('Not yet implemented.')
        # exit(2)


if __name__ == '__main__':
    main()
