from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy


def plot_id(r, c):
    """ Compute a unique 1-based index for each plot based on its 1-based row and column"""
    return 2 * (r - 1) + c


def create_bibiplot(d1, d2):
    """
    1. plot component 1 against component 2 and component 3 for modality 1 and modality 2
    2. color by cluster (or t-type) eventually
    3. plot lines for genes or electrophys features where length is correlation with latent space (max val 1.0)


    :return:
    """

    titles = ('Transcriptomic space', 'Electrophysiological space')


    fig = make_subplots(rows=2, cols=2,
                        column_titles=titles)

    unit = 1  # radius of circle

    d1 = d1[:, :3]
    d2 = d2[:, :3]

    for i in range(3):
        d1[:, i] = d1[:, i] / np.max(np.abs(d1[:, i])) * 0.707
        d2[:, i] = d2[:, i] / np.max(np.abs(d2[:, i])) * 0.707


    # Scatter plots
    for r in [1, 2]:
        c = 1
        fig.add_trace(go.Scatter(x=d1[:, 0], y=d1[:, r], mode='markers',
                                 marker={'size': 2},
                                 xaxis=f'x{plot_id(r, c)}',
                                 yaxis=f'y{plot_id(r, c)}',
                                 showlegend=False),
                      row=r, col=c)
        c = 2
        fig.add_trace(go.Scatter(x=d2[:, 0], y=d2[:, r], mode='markers',
                                 marker={'size': 2},
                                 xaxis=f'x{plot_id(r, c)}',
                                 yaxis=f'y{plot_id(r, c)}',
                                 showlegend=False),
                      row=r, col=c)

    for r in [0, 1]:
        for c in [0, 1]:
            fig.add_shape(type='circle',
                          line={'color': 'Black', 'width': 1},
                          #line_color="Black",
                          xref='x', yref='y',
                          x0=-unit, y0=-unit, x1=unit, y1=unit,
                          row=r, col=c)

    for r in [1, 2]:
        c = 1
        fig.update_yaxes(title_text=f'Component {r+1}', row=r, col=c,
                         scaleanchor=f'x{plot_id(r, c)}', scaleratio=1,
                         showticklabels=False)
        c = 2
        fig.update_yaxes(scaleanchor=f'x{plot_id(r, c)}', scaleratio=1, row=r, col=c,
                         showticklabels=False)
    for c in [1, 2]:
        fig.update_xaxes(title_text=f'Component 1', row=2, col=c,
                         showticklabels=False)
        fig.update_xaxes(row=1, col=c,
                         showticklabels=False)


    return fig



def create_bibiplot_zzz(dataset):
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