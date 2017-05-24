# coding: utf-8

import pandas as pd
import numpy as np
import re
import pickle
import gensim
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from gensim import corpora
from gensim import models
from gensim.corpora.dictionary import Dictionary

from gensim.parsing.preprocessing import STOPWORDS
from time import time

import string
import logging

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    action='store',
                    choices=['FP4', 'FP5', 'FP6', 'FP7', 'H2020'],
                    help='input dataset')

args = parser.parse_args()

df_name = args.dataset

df = pd.read_pickle('dfs/' + df_name)
df1 = df[['title','objective']]
df1 = df1.dropna(how='any')
df1['merged'] = df1['title'] + ' ' + df1['objective']

num_words = df1['merged'].str.lower().str.split().apply(lambda x: len(x)).sum()
print(num_words, ' read')

objectives = df1['merged']

RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

objectives = objectives.str.lower().str.replace('%l', '').str.replace(RE_PUNCTUATION, ' ')
# objectives.head(2)

objectives_split = objectives.str.strip().str.split()
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if len(token) > 2])
# objectives_split.head(2)

list_stopwords = ['will', 'develop', 'development', 'project', 'research', 'new', 'use', 'europe', 'european', 'based']
if df_name == 'FP4':
    list_stopwords.append('des')
additional_stopwords = set(list_stopwords)
stopwords = set(STOPWORDS) | additional_stopwords

objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])

objectives_dictionary = Dictionary(objectives_split)

class ObjectivesCorpus(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return iter(self.input)
    def __calc_corpus_size__(self):
        logging.info('Calculating corpus size')
        self.length = 0
        self.num_words = 0
        for doc in self.get_texts():
            self.length += 1
            self.num_words += len(doc)
    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            self.__calc_corpus_size__()
        return self.length
    def __str__(self):
        if 'num_words' not in self.__dict__:
            self.__calc_corpus_size__()
        return (str(self.length) + ' documents, ' + str(self.num_words)
                + ' words')
            

objectives_corpus = ObjectivesCorpus(objectives_split)

print(objectives_corpus)

t0 = time()
iterations = 300
if df_name == 'FP7':
    iterations = 400
elif df_name == 'H2020':
    iterations = 250
lda = gensim.models.ldamodel.LdaModel(corpus=objectives_corpus, 
                                      id2word=objectives_dictionary, 
                                      num_topics=10,
                                      iterations = iterations,
                                      random_state=np.random.seed(42))
print("done in %0.3fs." % (time() - t0))

lda.print_topics(10)

for t in range(lda.num_topics):
    words = dict(lda.show_topic(t, 15))
    elements = WordCloud(background_color='white').fit_words(words)
    plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    t = t + 1
    plt.title("Topic #" + str(t))
    # plt.show()
    plt.savefig('EU' + df_name + '_topic' + str(t) + '.png')
plt.close()
