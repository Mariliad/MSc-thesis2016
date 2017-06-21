# coding: utf-8

import pandas as pd
import numpy as np
import gensim
import argparse
import pickle
import scipy.stats as stats
import dit

from gensim import corpora, models, similarities, matutils
from gensim.corpora.dictionary import Dictionary
from dit.divergences import jensen_shannon_divergence
from scipy.stats import entropy
from numpy.linalg import norm

parser = argparse.ArgumentParser()

parser.add_argument('dataset',
                    action='store',
                    choices=['FP4', 'FP5', 'FP6', 'FP7', 'H2020'],
                    help='input dataset')

args = parser.parse_args()

df = args.dataset

# Load the LDA model of the two datasets
ldaEU = gensim.models.ldamodel.LdaModel.load('./eu_gensim/lda_saved/lda_eu'+ df +'.model')
ldaUSA = gensim.models.ldamodel.LdaModel.load('./usa_gensim/lda_saved/lda_usa'+ df +'.model')

def get_topic_vectors(topic_eu_id, topic_usa_id):
	# get the words and their probabilities per topic
	get_topic_terms_eu = ldaEU.show_topic(topic_eu_id, topn=ldaEU.num_terms)
	get_topic_terms_usa = ldaUSA.show_topic(topic_usa_id, topn=ldaUSA.num_terms)

	# keep the words in a list as a dictionary
	eu_list = [x[0] for x in get_topic_terms_eu]
	usa_list = [x[0] for x in get_topic_terms_usa]

	# add the words that don't exist in one of them, with probability = 0.0
	for i in get_topic_terms_eu:
		if i[0] not in usa_list:
			get_topic_terms_usa.append((i[0], 0.0))

	for j in get_topic_terms_usa:
		if j[0] not in eu_list:
			get_topic_terms_eu.append((j[0], 0.0))

	# sort the list of the tuples (word, probability) in alphabetic order
	get_topic_terms_eu = sorted(get_topic_terms_eu, key=lambda x: x[0])
	get_topic_terms_usa = sorted(get_topic_terms_usa, key=lambda x: x[0])

	# get only the probabilities as vectors
	topicEUvec = [x[1] for x in get_topic_terms_eu]
	topicUSAvec = [x[1] for x in get_topic_terms_usa]


	topicEUvec = dit.ScalarDistribution([x[0] for x in get_topic_terms_eu], [x[1] for x in get_topic_terms_eu])
	topicUSAvec = dit.ScalarDistribution([x[0] for x in get_topic_terms_usa], [x[1] for x in get_topic_terms_usa])

	return topicEUvec, topicUSAvec

# Function for calculating the Jensen Shannon Divergence
def JSD(P, Q):
    # _P = P / norm(P, ord=1)
    # _Q = Q / norm(Q, ord=1)
    _P = np.array(P)
    _Q = np.array(Q)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
# check if it computes the KL

# print JSD(topicEUvec, topicUSAvec)

print 'Framework Programme', df
print

eu = [i for i in range(10)]
usa = [i for i in range(10)]
dfJSD = pd.DataFrame(index=eu, columns=usa)


# for eu_topic in range(ldaEU.num_topics):
# 	for usa_topic in range(ldaUSA.num_topics):
# 		topicEUvec, topicUSAvec = get_topic_vectors(eu_topic, usa_topic)
		# print 'EU topic: ', eu_topic, ' USA topic: ', usa_topic,
		# print ' --> JSD = ', JSD(topicEUvec, topicUSAvec)
		# dfJSD.set_value(index=eu_topic, col=usa_topic, value=JSD(topicEUvec, topicUSAvec))
		# dfJSD.set_value(index=eu_topic, col=usa_topic, value=jensen_shannon_divergence([topicEUvec, topicUSAvec]))

# dfJSD.to_pickle('compared_FPs/JSD_' + df)

# print dfJSD.head(1)

# # Caluculates symmetric Kullback-Leibler divergence.
# def symmetric_kl_divergence(p, q):
# 	return numpy.sum([stats.entropy(p, q), stats.entropy(q, p)])


topicEUvec, topicUSAvec = get_topic_vectors(0, 0)
# print JSD(topicEUvec, topicUSAvec)
print jensen_shannon_divergence([topicEUvec, topicUSAvec])