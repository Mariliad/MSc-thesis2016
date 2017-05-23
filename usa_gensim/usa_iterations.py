# coding: utf-8

import pandas as pd
import numpy as np
import re
import pickle
import gensim
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from gensim import corpora
from gensim import models
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import STOPWORDS

from time import time

import string
import logging

# iterations_list = [5000, 6000, 7000, 7500]

# usa_dataset_list = ['FP4', 'FP5', 'FP6', 'FP7', 'H2020']
# fp4 >= 3500
# fp5 >= 3700
# fp6: very close to 4000
# fp7 >4500
# H2020 = 5000
df_name = 'H2020'

df = pd.read_pickle('pickle_data/usa' + df_name)
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

objectives_dictionary = Dictionary(objectives_split)

# check the method's 'filter_extremes' parameters
objectives_dictionary.filter_extremes(no_below=5)

class ObjectivesCorpus(object):
    def __init__(self, documents, dictionary):
        self.documents = documents
        self.dictionary = dictionary
    def __iter__(self):
        for document in self.documents:
            yield self.dictionary.doc2bow(document)

objectives_corpus = ObjectivesCorpus(objectives_split, objectives_dictionary)

t0 = time()
# random_state=np.random.seed(42)
iterations = 7500
# for i in range(8):
lda = gensim.models.ldamodel.LdaModel(corpus = objectives_corpus, 
                                        id2word = objectives_dictionary, 
                                        num_topics = 10,
                                        iterations = iterations,
                                        random_state=np.random.seed(42))

    # with open('usa_'+ df_name + '_topics/topics' + df_name + '_' + str(iterations) + '_' + str(i) + ".txt", "w") as text_file:
    #     text_file.write(str(lda.print_topics(num_topics=10, num_words=15)))
    #     text_file.close()

print("done in %0.3fs." % (time() - t0))

for t in range(lda.num_topics):
    words = dict(lda.show_topic(t, 15))
    elements = WordCloud(background_color='white').fit_words(words)
    plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    t = t + 1
    plt.title("Topic #" + str(t) + "with " + str(iterations) + ' iterations')
    # plt.savefig('USA' + df_name + '_topic' + str(t) + '_' + str(i) + '_' + str(iterations) + '.png')
    plt.savefig('usa_'+ df_name + '_wordclouds/USA' + df_name + '_topic' + str(t) + '_' + str(iterations) + '.png')
    plt.close()

