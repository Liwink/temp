#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as ply

print('load data and preprocess')
questions = pd.read_csv('./question_train_word.csv')
questions_topics = questions.topics.apply(lambda s: s.split(','))
question_titles = questions.title.astype('U').apply(lambda s: s.split(','))

from collections import Counter

question_words = [w for ws in question_titles for w in ws]
word_counter = Counter(question_words)
word_most_common = word_counter.most_common()

topic_count = Counter([t for ts in questions_topics for t in ts])
topic_most_common = np.array(topic_count.most_common())

def tag_index(i, length=len(questions_topics)):
    tag = topic_most_common[i][0]
    y_index = []
    for topics in questions_topics[:int(length)]:
        if tag in topics:
            y_index.append(1)
        else:
            y_index.append(0)
    return y_index

y_most_common = tag_index(0)

print('TfidfVectorizer')
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_df=0.85, min_df=5, ngram_range=(1,2))
X_train = vect.fit_transform(questions.title.astype('U'))

print('SGD')

from sklearn.linear_model import SGDClassifier
from tqdm import tqdm


def predict(clf_list, length=2999967, start=0, word_tfidf=X_train):
    predict_list = []
    for clf in tqdm(clf_list):
        try:
            result = [1-j[0] for j in clf.predict_proba(word_tfidf[start:start+length])]
        except:
            result = np.zeros(length)
        predict_list.append(result)
    return predict_list

topic_map = {i: v[0] for i, v in enumerate(topic_most_common)}

print('multiprocessing')

import multiprocessing


from sklearn.externals import joblib

def fit(args):
    i, clf = args
    y = tag_index(i)
    clf.fit(X_train, y)
    joblib.dump(clf, 'sgd_classifier_v3/sgd_{}.pkl'.format(i))
    return i

sgd_train_list = [(i, SGDClassifier(loss="modified_huber", penalty="l2", n_jobs=-1, alpha=1e-7, n_iter=1)) for i in range(1999)]

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
