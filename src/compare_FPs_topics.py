# coding: utf-8

import pandas as pd
import numpy as np
import gensim
import argparse
import scipy.stats as stats

from gensim import corpora, models, similarities, matutils
from gensim.corpora.dictionary import Dictionary
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
ldaEU = gensim.models.ldamodel.LdaModel.load('./eu/lda_saved/lda_eu'
                                             + df +'.model')
ldaUSA = gensim.models.ldamodel.LdaModel.load('./usa/lda_saved/lda_usa'
                                              + df +'.model')

def get_topic_vectors(topic_eu_id, topic_usa_id):

    # get the words and their probabilities for the topic
    eu_topic_terms_tuples = ldaEU.show_topic(topic_eu_id,
                                             topn=ldaEU.num_terms)
    us_topic_terms_tuples = ldaUSA.show_topic(topic_usa_id,
                                              topn=ldaUSA.num_terms)

    # keep the words in set
    eu_topic_terms = { x[0] for x in eu_topic_terms_tuples }
    us_topic_terms = { x[0] for x in us_topic_terms_tuples }

    # add the words that don't exist in one of them, with probability = 0.0
    for eu_missing in us_topic_terms - eu_topic_terms:
        eu_topic_terms_tuples.append((eu_missing, 0.0))
                    
    for us_missing in eu_topic_terms - us_topic_terms:
        us_topic_terms_tuples.append((us_missing, 0.0))

    # sort the list of the tuples (word, probability) in alphabetic order
    eu_topic_terms_tuples = sorted(eu_topic_terms_tuples, key=lambda x: x[0])
    us_topic_terms_tuples = sorted(us_topic_terms_tuples, key=lambda x: x[0])

    # get only the probabilities as vectors
    eu_topic_probs = [ x[1] for x in eu_topic_terms_tuples ]
    usa_topic_probs = [ x[1] for x in us_topic_terms_tuples ]

    return eu_topic_probs, usa_topic_probs


# Function for calculating the Jensen Shannon Divergence
def calc_jsd(P, Q):
    _P = np.array(P)
    _Q = np.array(Q)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

print('Framework Programme ' +  df)

eu = [('eu' + str(i)) for i in range(10)]
usa = [('usa' + str(i)) for i in range(10)]
dfJSD = pd.DataFrame(index=eu, columns=usa)

for eu_topic in range(ldaEU.num_topics):
	for usa_topic in range(ldaUSA.num_topics):
		eu_topic_probs, us_topic_probs = get_topic_vectors(eu_topic, usa_topic)
		dfJSD.set_value(index=('eu'+ str(eu_topic)), col=('usa'+ str(usa_topic)), value=calc_jsd(eu_topic_probs, us_topic_probs))

# dfJSD.to_pickle('compared_FPs/JSD_' + df)
dfJSD.to_csv('compared_FPs/JSD_' + df + '.csv', sep = ';', encoding='utf-8')

print dfJSD.index
print dfJSD.columns
# eu_topic_probs, us_topic_probs = get_topic_vectors(0, 0)
# print(calc_jsd(eu_topic_probs, us_topic_probs))