#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from collections import Counter
from evaluate import evaluate
import sys

sgd_alpha = float(sys.argv[1:][0])
print(sgd_alpha)
tfidf_ngram_range = (1, 2)
filename = 'sgd_classifier_v4_cv/sgd_{}_{}_{}.pkl'

questions = pd.read_csv('./question_train_word.csv')
questions_topics = questions.topics.apply(lambda s: s.split(','))

topic_count = Counter([t for ts in questions_topics for t in ts])
topic_most_common = np.array(topic_count.most_common())

tag_index_dict = {}
stop_length = int(len(questions_topics) / 10)

topic_map = {i: v[0] for i, v in enumerate(topic_most_common)}


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


def predict(clf_list, word_tfidf):
    predict_list = []
    for clf in clf_list:
        result = [1 - j[0] for j in clf.predict_proba(word_tfidf)]
        predict_list.append(result)
    return predict_list


def transform(predict_list):
    return np.array(predict_list).T


def top_five(predict_list):
    return np.array(predict_list).argsort()[-5:][::-1]


def transform_predict(predict_list):
    predict_list_t = transform(predict_list)
    result = []
    for p in predict_list_t:
        result.append(list(map(lambda x: topic_map[x], top_five(p))))
    return result

print('start: tfidf fit_transform')

vect = TfidfVectorizer(max_df=0.45, min_df=21, ngram_range=tfidf_ngram_range)
X_train = vect.fit_transform(questions.title.astype('U'))

print('start: loading')
sgd_list = []
for i in range(1999):
    sgd_list.append(joblib.load(filename.format(tfidf_ngram_range, sgd_alpha, i)))

print('start: predict')
predict_list = predict(sgd_list, X_train[-stop_length:])
result = transform_predict(predict_list)

print(evaluate(zip(result, questions_topics[-stop_length:])))
pd.Series(result).to_csv('sgd_result_v4_cv_2_{}.csv'.format(sgd_alpha), header=False, index=False, sep=' ')
