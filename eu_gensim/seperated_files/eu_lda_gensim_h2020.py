
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

dfH2020 = pd.read_pickle('dfs/df20')

df1 = dfH2020[['title','objective']]
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
                                              num_topics=10,
                                              iterations = 50)
print("done in %0.3fs." % (time() - t0))


print lda.print_topics(num_topics=10, num_words=15)
print

# for t in range(lda.num_topics):
#     words = dict(lda.show_topic(t, 15))
#     print words
#     elements = WordCloud(background_color='white').fit_words(words)
#     plt.figure()
#     plt.imshow(elements)
#     plt.axis("off")
#     plt.title("Topic #" + str(t))
#     # plt.show()
#     plt.savefig('euH2020_topic' + str(t) + '.png')
#     plt.close()

# [
# 0, u'0.012*"data" + 0.005*"policy" + 0.004*"climate" + 0.004*"management" + 0.004*"public" + 0.004*"information" + 0.004*"economic" + 0.004*"global" + 0.003*"food" + 0.003*"social" + 0.003*"innovative" + 0.003*"sustainable" + 0.003*"innovation" + 0.003*"impact" + 0.003*"market"'), 
# 1, u'0.006*"training" + 0.004*"health" + 0.004*"knowledge" + 0.004*"researchers" + 0.004*"technology" + 0.004*"innovation" + 0.004*"innovative" + 0.004*"scientific" + 0.003*"social" + 0.003*"support" + 0.003*"care" + 0.003*"science" + 0.003*"industry" + 0.003*"activities" + 0.003*"partners"'), 
# 2, u'0.012*"cell" + 0.011*"cells" + 0.008*"clinical" + 0.008*"cancer" + 0.007*"patients" + 0.005*"treatment" + 0.005*"disease" + 0.004*"tissue" + 0.004*"therapy" + 0.003*"novel" + 0.003*"therapeutic" + 0.003*"high" + 0.003*"technology" + 0.003*"patient" + 0.003*"stem"'), 
# 3, u'0.004*"novel" + 0.004*"role" + 0.004*"understanding" + 0.004*"disease" + 0.004*"study" + 0.004*"molecular" + 0.004*"mechanisms" + 0.003*"gene" + 0.003*"cancer" + 0.003*"genetic" + 0.003*"human" + 0.003*"genome" + 0.003*"studies" + 0.003*"specific" + 0.003*"identify"'), 
# 4, u'0.004*"control" + 0.003*"physics" + 0.003*"systems" + 0.003*"models" + 0.003*"time" + 0.003*"novel" + 0.003*"high" + 0.003*"model" + 0.003*"techniques" + 0.003*"methods" + 0.003*"water" + 0.003*"understanding" + 0.003*"field" + 0.003*"brain" + 0.002*"experimental"'), 
# 5, u'0.013*"market" + 0.012*"energy" + 0.007*"technology" + 0.006*"business" + 0.005*"production" + 0.005*"cost" + 0.004*"high" + 0.004*"solution" + 0.004*"product" + 0.004*"water" + 0.004*"innovative" + 0.004*"industry" + 0.004*"innovation" + 0.004*"process" + 0.003*"products"'), 
# 6, u'0.021*"innovation" + 0.010*"management" + 0.008*"smes" + 0.007*"capacity" + 0.006*"services" + 0.006*"sme" + 0.006*"support" + 0.005*"network" + 0.004*"instrument" + 0.004*"training" + 0.004*"international" + 0.004*"programme" + 0.004*"activities" + 0.003*"enhancing" + 0.003*"potential"'), 
# 7, u'0.004*"cell" + 0.003*"novel" + 0.003*"human" + 0.003*"high" + 0.003*"approach" + 0.003*"control" + 0.003*"mechanisms" + 0.003*"protein" + 0.003*"production" + 0.003*"process" + 0.003*"understanding" + 0.003*"brain" + 0.003*"functional" + 0.003*"molecular" + 0.003*"processes"'), 
# 8, u'0.006*"theory" + 0.006*"social" + 0.005*"study" + 0.003*"cultural" + 0.003*"analysis" + 0.003*"science" + 0.003*"field" + 0.003*"approach" + 0.003*"different" + 0.002*"understanding" + 0.002*"history" + 0.002*"political" + 0.002*"work" + 0.002*"imaging" + 0.002*"methods"'), 
# 9, u'0.008*"systems" + 0.007*"quantum" + 0.007*"high" + 0.006*"energy" + 0.005*"materials" + 0.004*"technology" + 0.004*"optical" + 0.003*"applications" + 0.003*"cost" + 0.003*"low" + 0.003*"devices" + 0.003*"time" + 0.003*"light" + 0.003*"performance" + 0.003*"novel"')]
