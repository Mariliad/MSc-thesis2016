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
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


dfFP6 = pd.read_pickle('pickle_data/usaFP6')
df1 = dfFP6[['title','objective']]
df1 = df1.dropna(how='any')
df1['merged'] = df1['title'] + ' ' + df1['objective']

objectives = df1['merged']


# remove numbers???
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

objectives = objectives.str.lower().str.replace(RE_PUNCTUATION, ' ')

objectives_split = objectives.str.strip().str.split()
objectives_split = objectives_split.apply(lambda tokens: [token for token in tokens if len(token) > 2])

# remove 'work' and 'based'?
additional_stopwords = set(['new', 'project', 'university', 'student', 'students', 'research', 'study', 
                            'program', 'development', 'study', 'studies', 'provide', 'use', 'work', 'based'])
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
lda = gensim.models.ldamodel.LdaModel(corpus=objectives_corpus, 
                                              id2word=objectives_dictionary, 
                                              num_topics=10)
print("done in %0.3fs." % (time() - t0))

# print lda.print_topics(10)

for t in range(lda.num_topics):
    # words = np.array(lda.show_topic(t, 10))
    words = dict(lda.show_topic(t, 15))
    elements = WordCloud(background_color='white').fit_words(words)
    # Wordcloud.draw(elements, "topic_%d.png" % (t), width=120, height=120)
    plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    plt.title("Topic #" + str(t))
    # plt.show()
    plt.savefig('usaFP6_topic' + str(t) + '.png')
    plt.close()


# In[ ]:
# [
# (0, u'0.010*"science" + 0.006*"education" + 0.005*"learning" + 0.005*"engineering" + 0.004*"data" + 0.004*"based" + 0.004*"school" + 0.004*"community" + 0.004*"information" + 0.004*"technology"'), 
# (1, u'0.005*"design" + 0.004*"based" + 0.004*"develop" + 0.004*"high" + 0.004*"data" + 0.004*"model" + 0.004*"understanding" + 0.003*"systems" + 0.003*"proposed" + 0.003*"materials"'), 
# (2, u'0.011*"data" + 0.008*"systems" + 0.005*"information" + 0.005*"network" + 0.004*"design" + 0.004*"networks" + 0.004*"time" + 0.004*"analysis" + 0.004*"applications" + 0.004*"models"'), 
# (3, u'0.011*"theory" + 0.007*"problems" + 0.005*"systems" + 0.004*"models" + 0.004*"work" + 0.004*"equations" + 0.004*"geometry" + 0.004*"field" + 0.004*"understanding" + 0.003*"important"'), 
# (4, u'0.007*"data" + 0.006*"climate" + 0.004*"models" + 0.004*"understanding" + 0.003*"time" + 0.003*"graduate" + 0.003*"support" + 0.003*"researchers" + 0.003*"work" + 0.003*"model"'), 
# (5, u'0.007*"science" + 0.007*"engineering" + 0.006*"undergraduate" + 0.005*"state" + 0.005*"education" + 0.005*"graduate" + 0.005*"technology" + 0.004*"chemistry" + 0.004*"high" + 0.004*"materials"'), 
# (6, u'0.006*"science" + 0.006*"data" + 0.005*"ice" + 0.004*"understanding" + 0.004*"graduate" + 0.003*"ocean" + 0.003*"high" + 0.003*"processes" + 0.003*"field" + 0.003*"work"'), 
# (7, u'0.008*"materials" + 0.006*"high" + 0.005*"properties" + 0.004*"systems" + 0.004*"applications" + 0.004*"based" + 0.004*"devices" + 0.003*"proposed" + 0.003*"phase" + 0.003*"technology"'), 
# (8, u'0.005*"data" + 0.004*"social" + 0.003*"understanding" + 0.003*"water" + 0.003*"processes" + 0.003*"chemistry" + 0.003*"work" + 0.002*"policy" + 0.002*"results" + 0.002*"state"'), 
# (9, u'0.005*"cell" + 0.005*"protein" + 0.004*"molecular" + 0.004*"species" + 0.004*"genes" + 0.004*"understanding" + 0.004*"proteins" + 0.004*"plant" + 0.004*"gene" + 0.004*"important"')]


# [
# (0, u'0.006*"species" + 0.005*"climate" + 0.004*"understanding" + 0.004*"data" + 0.004*"change" + 0.003*"changes" + 0.003*"important" + 0.003*"field" + 0.003*"conference" + 0.003*"environmental"'), 
# (1, u'0.007*"materials" + 0.007*"chemistry" + 0.004*"high" + 0.004*"state" + 0.004*"molecular" + 0.003*"organic" + 0.003*"chemical" + 0.003*"properties" + 0.003*"systems" + 0.003*"understanding"'), 
# (2, u'0.010*"theory" + 0.007*"problems" + 0.007*"data" + 0.005*"systems" + 0.005*"methods" + 0.005*"models" + 0.005*"model" + 0.005*"analysis" + 0.004*"applications" + 0.003*"important"'), 
# (3, u'0.015*"data" + 0.004*"information" + 0.004*"science" + 0.004*"materials" + 0.003*"systems" + 0.003*"understanding" + 0.003*"analysis" + 0.003*"dna" + 0.003*"including" + 0.003*"scientific"'), 
# (4, u'0.010*"science" + 0.005*"education" + 0.004*"school" + 0.004*"graduate" + 0.003*"community" + 0.003*"undergraduate" + 0.003*"social" + 0.003*"engineering" + 0.003*"understanding" + 0.003*"data"'), 
# (5, u'0.011*"science" + 0.006*"high" + 0.006*"materials" + 0.005*"engineering" + 0.005*"teachers" + 0.005*"education" + 0.004*"school" + 0.004*"undergraduate" + 0.004*"faculty" + 0.004*"mathematics"'), 
# (6, u'0.005*"science" + 0.005*"models" + 0.004*"high" + 0.004*"understanding" + 0.003*"graduate" + 0.003*"model" + 0.003*"east" + 0.003*"data" + 0.003*"field" + 0.003*"asia"'), 
# (7, u'0.005*"data" + 0.004*"high" + 0.004*"systems" + 0.004*"understanding" + 0.004*"flow" + 0.003*"proposed" + 0.003*"time" + 0.003*"analysis" + 0.003*"field" + 0.003*"models"'), 
# (8, u'0.004*"data" + 0.004*"understanding" + 0.004*"plant" + 0.004*"genes" + 0.003*"cell" + 0.003*"different" + 0.003*"gene" + 0.003*"species" + 0.003*"plants" + 0.003*"methods"'), 
# (9, u'0.007*"engineering" + 0.007*"design" + 0.006*"systems" + 0.006*"technology" + 0.005*"information" + 0.004*"high" + 0.004*"software" + 0.004*"data" + 0.004*"education" + 0.004*"performance"')]

