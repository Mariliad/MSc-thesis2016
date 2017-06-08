# coding: utf-8

import pandas as pd
import numpy as np
import re
import pickle
import gensim

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

df4 = pd.read_pickle('dfs/' + 'FP4')
df5 = pd.read_pickle('dfs/' + 'FP5')
df6 = pd.read_pickle('dfs/' + 'FP6')
df7 = pd.read_pickle('dfs/' + 'FP7')
df20 = pd.read_pickle('dfs/' + 'H2020')

def get_merged_docs(df):
    df = df[['title','objective']]
    df = df.dropna(how='any')
    df['merged'] = df['title'] + ' ' + df['objective']
    return df['merged']

objectives = pd.concat([get_merged_docs(df) for df in [df4, df5, df6, df7, df20] ])


RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

objectives = objectives.str.lower().str.replace('%l', '').str.replace(RE_PUNCTUATION, ' ')

objectives_split = objectives.str.strip().str.split()
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if len(token) > 2])
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if not(token.isdigit())])
objectives_split.head(2)


list_stopwords = ['data','will', 'develop', 'development', 'project', 'research', 'new', 'use', 'europe', 'european', 'based']
# if dfname == 'FP4':
#     list_stopwords.append('des')

additional_stopwords = set(list_stopwords)
stopwords = set(STOPWORDS) | additional_stopwords

objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])


frequency = defaultdict(int)
for text in objectives_split:
    for token in text:
        frequency[token] += 1

objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if (frequency[token] > 5)])

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

t0 = time()
lda = gensim.models.ldamodel.LdaModel(corpus = objectives_corpus, 
                                        id2word = objectives_dictionary, 
                                        num_topics = 10,
                                        random_state = np.random.seed(42),
                                        iterations = 7000)

print("done in %0.3fs." % (time() - t0))
lda.show_topics()

for t in range(lda.num_topics):
    words = dict(lda.show_topic(t, 20))
    elements = WordCloud(background_color='white', width=300, height=180, max_font_size=36, colormap='winter', prefer_horizontal=1.0).fit_words(words)
    plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    t = t + 1
    plt.title("Topic #" + str(t))
    # plt.savefig('EU' + df_name + '_topic' + str(t) + '_' + str(iterations) + '.png')
    plt.savefig('EU' + '_topic' + str(t) + '_passes1' + '.png')
    plt.close()



# # ************************************
# # get_document_topics ???