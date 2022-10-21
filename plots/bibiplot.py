from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy
import plotly.express as px
from application.settings import color_types


def plot_id(r, c):
    """ Compute a unique 1-based index for each plot based on its 1-based row and column"""
    return 2 * (r - 1) + c


def create_bibiplot1x2(d1, d2, x_col, y_col, dataset, color):
    """
    Create 1x2 bibiplot
    :param d1: projection of dataset1 into latent space
    :param d2: projection of dataset2 into latent space
    :param x_col: latent space component used as the x plotting dimension
    :param y_col: latent space component used as the y plotting dimension
    :return:
    """


    # from bibiplot_motor.ipynb
    geneExpr = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_filtered.csv', index_col=0)
    geneExpr = pd.DataFrame(geneExpr, dtype='int')
    count = scipy.sparse.csr_matrix(geneExpr).T  # transpose to put cells in rows, genes in columns
    seqDepth = np.sum(count, axis=1)  # sum gene enrichments across columns
    seqDepth = np.array(seqDepth)  # convert from matrix to vector
    X = np.log10(count / seqDepth * np.median(seqDepth) + 1)  # normalize  (not clear that multiplying by the median adds much value)
    X = np.array(X)
    X = X - X.mean(axis=0)
    X = X / np.std(X, axis=0)

    labels_X = geneExpr.index.array  # gene names

    ephysY = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_filtered.csv')
    if type(ephysY.iloc[0, 0]) == str:
        # drop the first column if it contains strings (i.e., presumably cell names)
        ephysY.drop(columns=ephysY.columns[0], inplace=True)

    ephysY_name = ephysY.columns.tolist()

    Y = ephysY.to_numpy() - np.mean(ephysY.to_numpy(), axis=0)
    Y = Y / np.std(Y, axis=0)
    labels_Y = ephysY_name

    Zx = d1[:, (x_col, y_col)]
    Zx = Zx / np.std(Zx, axis=0)

    Zy = d2[:, (x_col, y_col)]
    Zy = Zy / np.std(Zy, axis=0)

    Rho_x = np.corrcoef(np.concatenate((Zx, X), axis=1), rowvar=False)[2:, :2]
    Rho_y = np.corrcoef(np.concatenate((Zy, Y), axis=1), rowvar=False)[2:, :2]


    titles = ('Transcriptomic space', 'Electrophysiological space')


    fig = make_subplots(rows=1, cols=2,
                        column_titles=titles,
                        shared_xaxes=True,
                        shared_yaxes=True)

    unit = 1  # radius of circle

    # Select only the columns that will be plotted
    d1 = d1[:, [x_col, y_col]]
    d2 = d2[:, [x_col, y_col]]

    # Scale to be contained within the unit circle
    for i in range(2):
        d1[:, i] = d1[:, i] / np.max(np.abs(d1[:, i])) * 0.707
        d2[:, i] = d2[:, i] / np.max(np.abs(d2[:, i])) * 0.707

    # Convert category to a color code
    categories = sorted(list(set(color)))
    color_dict = {category: px.colors.qualitative.Plotly[i] for i, category in enumerate(categories)}
    #colors = [color_dict[c] for c in color]

    # Scatter plots
    for r in [1]:
        for category in categories:
            subset = list(color == category)
            c = 1
            fig.add_trace(go.Scatter(x=d1[subset, 0], y=d1[subset, 1], mode='markers',
                                     marker={'size': 2, 'color': color_dict[category]},
                                     xaxis=f'x{plot_id(r, c)}',
                                     yaxis=f'y{plot_id(r, c)}',
                                     showlegend=False,
                                     hoverinfo="none",
                                     name=category
                                     ),
                          row=r, col=c)
            c = 2
            fig.add_trace(go.Scatter(x=d2[subset, 0], y=d2[subset, 1], mode='markers',
                                     marker={'size': 2, 'color': color_dict[category]},
                                     xaxis=f'x{plot_id(r, c)}',
                                     yaxis=f'y{plot_id(r, c)}',
                                     showlegend=True,
                                     hoverinfo="none",
                                     name=category
                                     ),
                          row=r, col=c)



        # c = 1
        # fig.add_trace(go.Scatter(x=d1[:, 0], y=d1[:, 1], mode='markers',
        #                          marker={'size': 2, 'color': colors},
        #                          xaxis=f'x{plot_id(r, c)}',
        #                          yaxis=f'y{plot_id(r, c)}',
        #                          showlegend=False,
        #                          hoverinfo="none"
        #                          ),
        #               row=r, col=c)
        # c = 2
        # fig.add_trace(go.Scatter(x=d2[:, 0], y=d2[:, 1], mode='markers',
        #                          marker={'size': 2, 'color': colors},
        #                          xaxis=f'x{plot_id(r, c)}',
        #                          yaxis=f'y{plot_id(r, c)}',
        #                          showlegend=False,
        #                          hoverinfo="none"),
        #               row=r, col=c)

    # Circles
    for r in [0]:
        for c in [0, 1]:
            fig.add_shape(type='circle',
                          line={'color': 'Black', 'width': 1},
                          #line_color="Black",
                          xref='x', yref='y',
                          x0=-unit, y0=-unit, x1=unit, y1=unit,
                          row=r, col=c)


    # Lines (correlations between genes/features and the latent space)
    label_y_offset = 8  # pixels
    for c, (Z, F, labels, Rho) in enumerate(zip([Zx, Zy], [X, Y], [labels_X, labels_Y], [Rho_x, Rho_y])):
        labels = np.array(labels)
        sig_idx = []
        for i in range(F.shape[1]):
            if np.sqrt(np.sum(Rho[i,:]**2)) > .6:
                sig_idx.append(i)
                fig.add_shape(type='line',
                              xref='x', yref='y',
                              row=1, col=c + 1,
                              x0=0, y0=0,
                              x1=Rho[i, 0], y1=Rho[i, 1],
                              line={'color': 'black', 'width': 1})

                fig.add_annotation(x=Rho[i, 0],
                                   y=Rho[i, 1],
                                   yshift=label_y_offset if Rho[i, 1] > -0.1 else -label_y_offset,
                                   text=labels[i],
                                   showarrow=False,
                                   row=1, col=c + 1)

        # Add labels to lines
        if False:
            fig.add_trace(go.Scatter(
                x=Rho[sig_idx, 0], y=Rho[sig_idx, 1],
                text=np.array(labels)[sig_idx],
                mode='text',
                showlegend=False,
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=c + 1)

    # Axis labels, etc
    for r in [1]:
        c = 1
        fig.update_yaxes(title_text=f'Component {y_col + 1}', row=r, col=c,
                         scaleanchor=f'x{plot_id(r, c)}', scaleratio=1,
                         showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)
        c = 2
        fig.update_yaxes(scaleanchor=f'x{plot_id(r, c)}', scaleratio=1, row=r, col=c,
                         showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)
    for c in [1, 2]:
        fig.update_xaxes(title_text=f'Component {x_col + 1}', row=1, col=c,
                         showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)
        fig.update_xaxes(row=1, col=c,
                         showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)

    # This appears to link x and y axes so both plots have same level of pan/zoom
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig.update_layout(plot_bgcolor='white',
                      legend={'itemsizing': 'constant', 'title': color_types[color.name]})

    return fig


def create_bibiplot2x2(d1, d2):
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