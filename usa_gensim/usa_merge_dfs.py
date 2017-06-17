# coding: utf-8

import pandas as pd
import numpy as np
import re
import pickle
import gensim
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from time import time
from wordcloud import WordCloud
from gensim import corpora
from gensim import models
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
from gensim.parsing.preprocessing import STOPWORDS

import string
import logging


dfusa = pd.read_pickle('pickle_data/dfUSA')

print dfusa[dfusa['year'].astype(str).str.contains('2017')]

RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

dfusa['merged'] = dfusa['merged'].str.lower().str.replace(RE_PUNCTUATION, ' ')

dfusa['merged'] = dfusa['merged'].str.strip().str.split()
dfusa['merged'] = dfusa['merged'].apply(lambda tokens: [token for token in tokens if len(token) > 2])
dfusa['merged'] = dfusa['merged'].apply(lambda tokens: [token for token in tokens if not(token.isdigit())])

list_stopwords = ['data', 'work', 'based', 'new', 'project', 'university', 'student', 'students', 'research', 
                'study', 'program', 'development', 'study', 'studies', 'provide', 'use', 'understanding', 'important',
                'support', 'proposed']

additional_stopwords = set(list_stopwords)
stopwords = set(STOPWORDS) | additional_stopwords

dfusa['merged'] = dfusa['merged'].apply(lambda tokens: [token for token in tokens if token not in stopwords])

frequency = defaultdict(int)
for text in dfusa['merged']:
    for token in text:
        frequency[token] += 1

dfusa['merged'] = dfusa['merged'].apply(lambda tokens: [token for token in tokens if (frequency[token] > 5)])

objectives_dictionary = Dictionary(dfusa['merged'])

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
            

objectives_corpus = ObjectivesCorpus(dfusa['merged'])

iterations = 8000
# passes = 10
t0 = time()
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
lda = gensim.models.ldamodel.LdaModel(corpus = objectives_corpus, 
                                        id2word = objectives_dictionary, 
                                        num_topics = 10,
                                        iterations = iterations,
                                        random_state=np.random.seed(42))

print("done in %0.3fs." % (time() - t0))

for t in range(lda.num_topics):
    words = dict(lda.show_topic(t, 15))
    elements = WordCloud(background_color='white', width=300, height=180, max_font_size=36, colormap='winter', prefer_horizontal=1.0).fit_words(words)
    plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    plt.title("Topic #" + str(t))
    # plt.savefig('USA' + df_name + '_topic' + str(t) + '_' + str(i) + '_' + str(iterations) + '.png')
    plt.savefig('usa_all_wordclouds/USA' + '_topic' + str(t) + '_' + str(iterations) + '_all_docs.png')
    plt.close()

def get_top_probability(doc):
    topic_doc_list = lda.get_document_topics(objectives_dictionary.doc2bow(doc))
    top_topic = sorted(topic_doc_list,key=lambda x: x[1], reverse=True)[0]
    return top_topic[1]

def get_top_topic(doc):
    topic_doc_list = lda.get_document_topics(objectives_dictionary.doc2bow(doc))
    top_topic = sorted(topic_doc_list,key=lambda x: x[1], reverse=True)[0]
    return top_topic[0]

# for doc in dfusa['merged']:
dfusa['top_prob_topic'] = dfusa['merged'].apply(lambda docs: get_top_probability(docs))
dfusa['top_topic'] = dfusa['merged'].apply(lambda docs: get_top_topic(docs))

print dfusa.shape

dfusa.to_pickle('pickle_data/df_usa_topics')