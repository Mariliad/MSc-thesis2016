
# coding: utf-8

# In[1]:

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
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


dfFP7 = pd.read_pickle('pickle_data/usaFP7')

df1 = dfFP7[['title','objective']]
df1 = df1.dropna(how='any')
df1['merged'] = df1['title'] + ' ' + df1['objective']

objectives = df1['merged']

# remove numbers???
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

objectives = objectives.str.lower().str.replace(RE_PUNCTUATION, ' ')

# In[8]:

objectives_split = objectives.str.strip().str.split()
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if len(token) > 2])


# In[9]:

additional_stopwords = set(['new', 'project', 'university', 'student', 'students', 'research', 'study', 
                            'program', 'development', 'study', 'studies', 'provide', 'use'])
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
    plt.savefig('usaFP7_topic' + str(t) + '.png')
    plt.close()

# [
# (0, u'0.014*"species" + 0.008*"plant" + 0.007*"biology" + 0.006*"genetic" + 0.006*"understanding" + 0.006*"biological" + 0.005*"gene" + 0.005*"plants" + 0.005*"genes" + 0.005*"training"'), 
# (1, u'0.015*"theory" + 0.011*"problems" + 0.009*"models" + 0.009*"methods" + 0.007*"systems" + 0.007*"mathematical" + 0.006*"analysis" + 0.006*"applications" + 0.005*"physics" + 0.005*"computational"'), 
# (2, u'0.015*"science" + 0.009*"education" + 0.009*"engineering" + 0.007*"stem" + 0.006*"community" + 0.006*"support" + 0.006*"faculty" + 0.006*"graduate" + 0.006*"learning" + 0.006*"school"'), 
# (3, u'0.013*"social" + 0.007*"data" + 0.007*"policy" + 0.005*"economic" + 0.005*"information" + 0.004*"understanding" + 0.004*"science" + 0.004*"public" + 0.003*"political" + 0.003*"people"'), 
# (4, u'0.026*"data" + 0.007*"information" + 0.006*"language" + 0.005*"software" + 0.005*"analysis" + 0.005*"learning" + 0.005*"large" + 0.004*"work" + 0.004*"human" + 0.004*"tools"'), 
# (5, u'0.009*"systems" + 0.009*"design" + 0.008*"high" + 0.008*"energy" + 0.007*"applications" + 0.007*"materials" + 0.006*"technology" + 0.006*"based" + 0.006*"performance" + 0.005*"devices"'), 
# (6, u'0.009*"chemistry" + 0.008*"chemical" + 0.008*"materials" + 0.007*"molecular" + 0.006*"properties" + 0.006*"cell" + 0.005*"undergraduate" + 0.005*"high" + 0.005*"understanding" + 0.005*"surface"'), 
# (7, u'0.011*"microbial" + 0.009*"production" + 0.007*"oil" + 0.006*"bacteria" + 0.006*"disease" + 0.005*"treatment" + 0.005*"health" + 0.005*"environmental" + 0.004*"bacterial" + 0.004*"biomass"'), 
# (8, u'0.013*"climate" + 0.009*"water" + 0.008*"data" + 0.008*"ocean" + 0.007*"change" + 0.006*"model" + 0.005*"understanding" + 0.005*"models" + 0.005*"carbon" + 0.005*"impacts"'), 
# (9, u'0.008*"earth" + 0.006*"ice" + 0.005*"data" + 0.004*"mantle" + 0.004*"understanding" + 0.004*"processes" + 0.004*"time" + 0.004*"field" + 0.004*"high" + 0.004*"history"')]
