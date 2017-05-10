
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


# In[ ]:

dfh2020 = pd.read_pickle('pickle_data/usaH2020')

df1 = dfh2020[['title','objective']]
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
                            'program', 'development', 'study', 'studies', 'provide', 'use', 'based'])
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
    plt.savefig('usaH2020_topic' + str(t) + '.png')
    plt.close()



# (0, u'0.008*"brain" + 0.007*"ice" + 0.005*"ocean" + 0.005*"time" + 0.004*"sea" + 0.004*"arctic" + 0.004*"imaging" + 0.004*"health" + 0.004*"model" + 0.004*"proposed"'), 
# (1, u'0.010*"cell" + 0.010*"species" + 0.008*"cells" + 0.006*"biology" + 0.006*"biological" + 0.006*"protein" + 0.005*"genetic" + 0.005*"plant" + 0.005*"molecular" + 0.005*"understanding"'), 
# (2, u'0.008*"earth" + 0.006*"data" + 0.006*"processes" + 0.005*"understanding" + 0.004*"high" + 0.004*"surface" + 0.004*"models" + 0.004*"mass" + 0.003*"measurements" + 0.003*"wind"'), 
# (3, u'0.017*"materials" + 0.008*"high" + 0.008*"energy" + 0.007*"properties" + 0.005*"applications" + 0.005*"design" + 0.005*"chemical" + 0.004*"devices" + 0.004*"material" + 0.004*"chemistry"'), 
# (4, u'0.013*"climate" + 0.007*"change" + 0.006*"changes" + 0.006*"species" + 0.005*"understanding" + 0.004*"data" + 0.004*"ocean" + 0.004*"environmental" + 0.004*"ecological" + 0.004*"global"'), 
# (5, u'0.012*"theory" + 0.009*"problems" + 0.008*"physics" + 0.007*"methods" + 0.007*"quantum" + 0.007*"models" + 0.007*"systems" + 0.005*"mathematical" + 0.005*"analysis" + 0.005*"computational"'), 
# (6, u'0.027*"water" + 0.010*"production" + 0.008*"environmental" + 0.008*"carbon" + 0.006*"energy" + 0.005*"gas" + 0.005*"microbial" + 0.005*"organic" + 0.004*"food" + 0.004*"process"'), 
# (7, u'0.013*"science" + 0.012*"stem" + 0.010*"engineering" + 0.009*"education" + 0.007*"learning" + 0.007*"support" + 0.006*"faculty" + 0.006*"community" + 0.006*"conference" + 0.006*"graduate"'), 
# (8, u'0.015*"data" + 0.010*"systems" + 0.007*"design" + 0.007*"software" + 0.006*"network" + 0.005*"performance" + 0.005*"algorithms" + 0.005*"applications" + 0.005*"information" + 0.004*"control"'), 
# (9, u'0.020*"data" + 0.011*"social" + 0.006*"science" + 0.005*"understanding" + 0.005*"learning" + 0.005*"policy" + 0.005*"public" + 0.004*"community" + 0.004*"knowledge" + 0.004*"information"')]
