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

# df4 = pd.read_pickle('dfs/' + 'FP4')
# df5 = pd.read_pickle('dfs/' + 'FP5')
# df6 = pd.read_pickle('dfs/' + 'FP6')
# df7 = pd.read_pickle('dfs/' + 'FP7')
# df20 = pd.read_pickle('dfs/' + 'H2020')

# def get_merged_docs(df):
#     df = df[['title','objective', 'framework_programme']]
#     df = df.dropna(how='any')
#     df['merged'] = df['title'] + ' ' + df['objective']
#     df1 = df[['merged', 'framework_programme']]
#     return df1

# df_all = pd.concat([get_merged_docs(df) for df in [df4, df5] ])

# print objectives.columns
# print objectives.shape

# objectives.to_pickle('df_all')
# ######################################################3
df_all = pd.read_pickle('dfs/df_all')

# print df_all.columns
# print df_all.shape
# print df_all.head(3)
# print

RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

df_all['merged'] = df_all['merged'].str.lower().str.replace('%l', '').str.replace(RE_PUNCTUATION, ' ')

df_all['merged'] = df_all['merged'].str.strip().str.split()
df_all['merged'] = df_all['merged'].apply(lambda tokens: [token for token in tokens if len(token) > 2])
df_all['merged'] = df_all['merged'].apply(lambda tokens: [token for token in tokens if not(token.isdigit())])
df_all['merged'].head(2)


list_stopwords = ['data','will', 'develop', 'development', 'project', 'research', 'new', 'use', 'europe', 'european', 'based']
# if dfname == 'FP4':
#     list_stopwords.append('des')

additional_stopwords = set(list_stopwords)
stopwords = set(STOPWORDS) | additional_stopwords

df_all['merged'] = df_all['merged'].apply(lambda tokens: [token for token in tokens if token not in stopwords])


frequency = defaultdict(int)
for text in df_all['merged']:
    for token in text:
        frequency[token] += 1

df_all['merged'] = df_all['merged'].apply(lambda tokens: [token for token in tokens if (frequency[token] > 5)])

objectives_dictionary = Dictionary(df_all['merged'])


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
            

objectives_corpus = ObjectivesCorpus(df_all['merged'])

# print df_all['merged'].head(1)

t0 = time()
lda = gensim.models.ldamodel.LdaModel(corpus = objectives_corpus, 
                                        id2word = objectives_dictionary, 
                                        num_topics = 10,
                                        random_state = np.random.seed(42),
                                        iterations = 7000)

print("done in %0.3fs." % (time() - t0))


# for t in range(lda.num_topics):
#     words = dict(lda.show_topic(t, 20))
#     elements = WordCloud(background_color='white', width=300, height=180, max_font_size=36, colormap='winter', prefer_horizontal=1.0).fit_words(words)
#     plt.figure()
#     plt.imshow(elements)
#     plt.axis("off")
#     t = t + 1
#     plt.title("Topic #" + str(t))
#     # plt.savefig('EU' + df_name + '_topic' + str(t) + '_' + str(iterations) + '.png')
#     plt.savefig('EU' + '_topic' + str(t) + '_all_docs' + '.png')
#     plt.close()

# corpus_lda = lda[objectives_corpus]


# for doc in df_all['merged']:
#     print doc
#     topic_doc_list = lda.get_document_topics(objectives_dictionary.doc2bow(doc))
#     print topic_doc_list
#     top_topic = sorted(topic_doc_list,key=lambda x: x[1], reverse=True)[0]
#     print top_topic
#     print
def get_top_probability(doc):
    topic_doc_list = lda.get_document_topics(objectives_dictionary.doc2bow(doc))
    top_topic = sorted(topic_doc_list,key=lambda x: x[1], reverse=True)[0]
    return top_topic[1]

def get_top_topic(doc):
    topic_doc_list = lda.get_document_topics(objectives_dictionary.doc2bow(doc))
    top_topic = sorted(topic_doc_list,key=lambda x: x[1], reverse=True)[0]
    return top_topic[0]

# for doc in df_all['merged']:
df_all['top_prob_topic'] = df_all['merged'].apply(lambda docs: get_top_probability(docs))
df_all['top_topic'] = df_all['merged'].apply(lambda docs: get_top_topic(docs))

print df_all.shape

df_all.to_pickle('dfs/df_topics')