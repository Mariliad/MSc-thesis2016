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

