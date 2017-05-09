
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

dfFP5 = pd.read_pickle('dfs/df5')

df1 = dfFP5[['title','objective']]
df1 = df1.dropna(how='any')
df1['merged'] = df1['title'] + ' ' + df1['objective']

objectives = df1['merged']


# remove numbers???
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

objectives = objectives.str.lower().str.replace(RE_PUNCTUATION, ' ')
objectives.head(2)

# In[8]:

objectives_split = objectives.str.strip().str.split()
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if len(token) > 2])
objectives_split.head(2)


# In[9]:

additional_stopwords = set(['will', 'develop', 'development',
                            'project', 'research', 'new', 'use', 
                            'europe', 'european', 'based'])
stopwords = set(STOPWORDS) | additional_stopwords

objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])


# In[10]:

objectives_dictionary = Dictionary(objectives_split)


# In[11]:

class ObjectivesCorpus(object):
    def __init__(self, documents, dictionary):
        self.documents = documents
        self.dictionary = dictionary
    def __iter__(self):
        for document in self.documents:
            yield self.dictionary.doc2bow(document)


# In[12]:

objectives_corpus = ObjectivesCorpus(objectives_split, objectives_dictionary)


# In[13]:

t0 = time()
lda = gensim.models.ldamodel.LdaModel(corpus=objectives_corpus, 
                                              id2word=objectives_dictionary, 
                                              num_topics=10)
print("done in %0.3fs." % (time() - t0))


# print lda.print_topics(10)

for t in range(lda.num_topics):
    words = dict(lda.show_topic(t, 15))
    elements = WordCloud(background_color='white').fit_words(words)
    plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    plt.title("Topic #" + str(t))
    # plt.show()
    plt.savefig('euFP5_topic' + str(t) + '.png')
    plt.close()

