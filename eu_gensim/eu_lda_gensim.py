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



df_name = ''
while df_name not in ['FP4', 'FP5', 'FP6', 'FP7', 'H2020']:
	df_name =raw_input("EU data: Choose dataset (options: FP4, FP5, FP6, FP7, H2020): ")

df = pd.read_pickle('dfs/' + df_name)
df1 = df[['title','objective']]
df1 = df1.dropna(how='any')
df1['merged'] = df1['title'] + ' ' + df1['objective']

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

class ObjectivesCorpus(object):
    def __init__(self, documents, dictionary):
        self.documents = documents
        self.dictionary = dictionary
    def __iter__(self):
        for document in self.documents:
            yield self.dictionary.doc2bow(document)

objectives_corpus = ObjectivesCorpus(objectives_split, objectives_dictionary)


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

# print lda.print_topics(num_topics=10, num_words=15)

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


