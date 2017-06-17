import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

df = pd.read_pickle('pickle_data/df_usa_topics')
print df.head(2)

print df.columns
print df.shape

df['year'] = df['year'].astype('str')

sz = df.groupby(["top_topic", 'year']).size()
sz = sz.reset_index()
sz = sz.rename(columns = {0:'count'})

print 'sz'
print sz.shape
print sz.columns

print sz.head(2)

sz['normalized'] = np.where(sz['year']=='1994', 
                    (sz['count']/sz.loc[sz['year'] == '1994', 'count'].sum())*100, sz['count'])

years = ['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
			'2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
for year in years:
	sz['normalized'] = np.where(sz['year']==year, 
                    (sz['count']/sz.loc[sz['year'] == year, 'count'].sum())*100, sz['normalized'])

top_year = sz.pivot(index='year', columns='top_topic', values='normalized')

print
print top_year
print top_year
top_year.to_pickle('pickle_data/df_year_topic')