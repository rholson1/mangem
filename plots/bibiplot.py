from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import scipy


def create_bibiplot(dataset):
    efeatures_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_NMA.csv')
    efeatures_NMA['gmm_cluster'] = efeatures_NMA['gmm_cluster'].astype('string')
    Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

    geneExp_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_NMA.csv')
    geneExp_NMA['gmm_cluster'] = geneExp_NMA['gmm_cluster'].astype('string')
    Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))



    geneExpr = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_filtered.csv', index_col=0)
    geneExpr = pd.DataFrame(geneExpr, dtype='int')
    count = scipy.sparse.csr_matrix(geneExpr).T

    seqDepth = np.sum(count, axis=1)
    seqDepth = np.array(seqDepth)
    np.median(seqDepth)
    X = np.log10(count/(seqDepth) * np.median(seqDepth) + 1)
    X = np.array(X)
    X = X - X.mean(axis=0)
    X = X / np.std(X, axis=0)

    ephysY = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_filtered.csv', index_col=0)
    ephysY_name = ephysY.columns.tolist()

    Y = ephysY.to_numpy() - np.mean(ephysY.to_numpy(), axis=0)
    Y = Y / np.std(Y, axis=0)

    print('Shape of X:', X.shape, '\nShape of Y:', Y.shape)








    fig = make_subplots(rows=1, cols=2, subplot_titles=('Electrophys', 'Gene Expression'),
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]])