#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from collections import Counter
from sklearn.linear_model import SGDClassifier
import multiprocessing
import os.path
import sys

sgd_alpha = float(sys.argv[1:][0])
print(sgd_alpha)

questions = pd.read_csv('./question_train_word.csv')
questions_topics = questions.topics.apply(lambda s: s.split(','))

topic_count = Counter([t for ts in questions_topics for t in ts])
topic_most_common = np.array(topic_count.most_common())

tag_index_dict = {}


def tag_index(i):
    tag = topic_most_common[i][0]
    if tag in tag_index_dict:
        return tag_index_dict[tag]
    y_index = []
    for topics in questions_topics:
        if tag in topics:
            y_index.append(1)
        else:
            y_index.append(0)
    tag_index_dict[tag] = y_index
    return y_index

tfidf_ngram_range = (1, 2)


def fit_cv(args):
    i, clf = args
    filename = 'sgd_classifier_v4_cv/sgd_{}_{}_{}.pkl'.format(tfidf_ngram_range, sgd_alpha, i)
    if os.path.isfile(filename):
        return i

    y = tag_index(i)

    length = int(len(y) / 10)
    clf.fit(X_train[:-length], y[:-length])
    joblib.dump(clf, filename)
    return i

print('start: tfidf fit_transform')

vect = TfidfVectorizer(max_df=0.45, min_df=21, ngram_range=tfidf_ngram_range)
X_train = vect.fit_transform(questions.title.astype('U'))

sgd_train_list = [(i, SGDClassifier(loss="modified_huber", penalty="l2", n_jobs=-1, alpha=sgd_alpha, n_iter=1))
                  for i in range(1999)]

print('start: fit_cv')
pool = multiprocessing.Pool(processes=4)
pool.map(fit_cv, sgd_train_list)
