#!/usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import randomforest as rforest
import sys
import tensorflow as tf


def show_usage():
    print('Usage:\n\t[python] ./main.py <m> <n> <k>\n'
          '(Hint: try m=100, n=50 and k=50)', file=sys.stderr)
    exit(2)


m = 100  # keep the m most frequent words,
n = 50  # ignore the n most frequent ones,
k = 50  # and ignore the k least frequent ones

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        show_usage()
    m = int(args[0])
    n = int(args[1])
    k = int(args[2])

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
    num_words=m,
    skip_top=n
)

# retrieve word index and create it's inverse
word_to_index = tf.keras.datasets.imdb.get_word_index()
index_to_word = dict(
    (i + index_from, word) for (word, i) in word_to_index.items()
)

# update inverted word index to include start_char and oov_char
index_to_word[0] = '[PAD]'
index_to_word[start_char] = '[START]'
index_to_word[oov_char] = '[OOV]'

# include k into the mix
for i in range(k):
    index_to_word[len(word_to_index) + index_from - i] = '[K]'

# decode each review in the dataset
x_train = np.array([' '.join([index_to_word[idx] for idx in text])
                   for text in x_train])
x_test = np.array([' '.join([index_to_word[idx] for idx in text])
                   for text in x_test])

# vectorize each review using CountVectorizer
binary_vectorizer = CountVectorizer(binary=True)
x_train_bin = binary_vectorizer.fit_transform(x_train).toarray()
x_test_bin = binary_vectorizer.fit_transform(x_test).toarray()
vocab = np.array([key for key in binary_vectorizer.vocabulary_.keys()])

if __name__ == '__main__':
    print(f'Params & Options:\nm = {m}\nn = {n}\nk = {k}\nVocab size: '
          f'{vocab.shape}\n===')

    sel_alg = int(input('Select an algorithm:\n[1]: Naive Bayes \
            \n[2]: Random Forest\n[3]: Adaboost\nYour selection: '))
    if sel_alg > 3 or sel_alg < 1:
        print('Invalid selection.')
        exit(-1)
    elif sel_alg == 1:
        print('Not yet implemented.')
        exit(1)
    elif sel_alg == 2:
        numberOfTrees = int(input('Number of trees (positive integer): '))
        forest = rforest.RandomForest(numberOfTrees, vocab)
        print(f'Training a Random Forest with {numberOfTrees} tree(s)...')
        forest.fit(x_train_bin, y_train)
        print('Fitting the Random Forest to the test data...')
        result = forest.predict(x_test_bin)
        print(f'===\nExpected (y_test):\n{y_test}\nGot (result):\n{result}')
    elif sel_alg == 3:
        print('Not yet implemented.')
        # exit(2)
