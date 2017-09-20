Topic modeling on EU and the USA research projects
========================
> A MSc thesis for AUEB MSc in Computer science.

### Authors
*   Maria Iliadi
*   Panos Louridas

This repository stores all the files of the MSc thesis. Here we present code to obtain topics from the projects funded under the European Union Framework Programmes, and for project funded by the NSF in the United States. We use Latent Dirichlet allocation for topic modeling in projects' abstracts that funded from 1994 to 2017.

## Getting Started

The `src` folder consists of all the code for generating LDA topics. `eu` and `usa` folders consist of:
- `*_raw_data`: the raw data of each region, before the first preprocessing
- `*_data_load_per_FP.ipynb`: the IPython notebook for loading the raw data, get the abstracts and save them as CSV files in the `dataset` folder for later use. In the USA dataset, you need to run the notebook, in order to create the `dataset` folder and the CSV files.
- `dataset`: the data (project abstracts) after preprocessing. The projects are grouped in both regions based on the FP/year that they got funded.
- `*_iterations.py`: the code for generating the topics per FP, using [LDA](https://radimrehurek.com/gensim/models/ldamodel.html).
- `lda_saved`: folder that contains the saved LDA model per FP (after training).
- `*_figures`: the wordclouds of the topics per FP, saved as PNG files.


## Generating LDA topics

To produce the topics for each FP, run:

`python eu_iterations.py FP_name [OPTIONS]`
`python usa_iterations.py FP_name [OPTIONS]`

where:

- `FP_name` is one of the following: FP4, FP5, FP6, FP7, H2020
- `-i` or `--iterations`: the number of iterations for the LDA model. By default it's 7000 for the EU dataset and 8000 for the USA.

The outcome is a set of 10 topics pictured as wordclouds, which they display the 20 top words with the highest probability of belonging to the topic.


## Similarity of topics

`compare_FPs_topics.py`: in order to calculate the similarity of the topics between the two regions (per FP), we calculate the [JSD](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) for each pair of topics and save the results in a CSV file (in the `compared_FPs` folder).

To run the code:

python compare_FPs_topics.py FP_name

where `FP_name` is one of the following: FP4, FP5, FP6, FP7, H2020.


## Requirements

- Python version 2.7 or later
- pandas 0.20 or later
- numpy 1.12 or later
- scipy 0.19 or later
- gensim 2.1 or later
- matplotlib 2.0 or later
- wordcloud 1.3
- ipython 5.4 combined with jupyter
- seaborn 0.7
