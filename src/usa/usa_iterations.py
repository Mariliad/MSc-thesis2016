# coding: utf-8

import pandas as pd
import numpy as np
import re
import gensim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from gensim import corpora
from gensim import models
from gensim.corpora.dictionary import Dictionary

from collections import defaultdict
from gensim.parsing.preprocessing import STOPWORDS
from time import time

import string
import logging
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    action='store',
                    choices=['FP4', 'FP5', 'FP6', 'FP7', 'H2020'],
                    help='input dataset')

parser.add_argument('-i', '--iterations',
                    type=int,
                    default=8000,
                    help='number of iterations')

args = parser.parse_args()

df_name = args.dataset

print args

df = pd.read_csv('dataset/usa' + df_name + '.csv', sep=";")
df1 = df[['title','objective']]
df1 = df1.dropna(how='any')
df1['merged'] = df1['title'] + ' ' + df1['objective']

objectives = df1['merged']

RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

objectives = objectives.str.lower().str.replace(RE_PUNCTUATION, ' ')

objectives_split = objectives.str.strip().str.split()
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if len(token) > 2])
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if not(token.isdigit())])

list_stopwords = ['data', 'work', 'based', 'new', 'project', 'university', 'student', 'students', 'research', 
                'study', 'program', 'development', 'study', 'studies', 'provide', 'use', 'understanding', 'important',
                'support', 'proposed']

additional_stopwords = set(list_stopwords)
stopwords = set(STOPWORDS) | additional_stopwords

objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])


frequency = defaultdict(int)
for text in objectives_split:
    for token in text:
        frequency[token] += 1

objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if (frequency[token] > 5)])

objectives_dictionary = Dictionary(objectives_split)

print (objectives_dictionary)
print
# objectives_dictionary.filter_extremes(no_below=5)

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

print'Corpus size: ', (objectives_corpus)

t0 = time()

lda = gensim.models.ldamodel.LdaModel(corpus = objectives_corpus, 
                                        id2word = objectives_dictionary, 
                                        num_topics = 10,
                                        iterations = args.iterations,
                                        random_state=np.random.seed(42))

print("done in %0.3fs." % (time() - t0))



for t in range(lda.num_topics):
    words = dict(lda.show_topic(t, 15))
    elements = WordCloud(background_color='white', width=300, height=180, max_font_size=36, colormap='winter', prefer_horizontal=1.0).fit_words(words)
    plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    t = t + 1
    plt.title("Topic #" + str(t))
    # plt.savefig('USA' + df_name + '_topic' + str(t) + '_' + str(i) + '_' + str(iterations) + '.png')
    plt.savefig('usa_figures/usa_'+ df_name + '_wordclouds/USA' + df_name + '_topic' + str(t) + '_' + str(args.iterations) + '.png')
    plt.close()

# def get_top_topic(doc):
#     topic_doc_list = lda.get_document_topics(objectives_dictionary.doc2bow(doc))
#     top_topic = sorted(topic_doc_list,key=lambda x: x[1], reverse=True)[0]
#     return top_topic[0]

# df1['top_topic'] = objectives_split.apply(lambda docs: get_top_topic(docs))