from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy
import plotly.express as px
from app_main.constants import color_types, plot_font_size, plot_title_font_size
from operations.preprocessing import preprocess


def plot_id(r, c):
    """ Compute a unique 1-based index for each plot based on its 1-based row and column"""
    return 2 * (r - 1) + c


def normalize_raw_df(data):
    total = np.sum(data, axis=1)  # sum measures across columns
    #total = np.array(total)  # convert from matrix to vector
    X = np.log10(data / total[:, None] * np.median(total) + 1)
    X = np.array(X)
    X = X - X.mean(axis=0)
    X = X / np.std(X, axis=0)
    return X


def create_bibiplot1x2(data_1, data_2, d1, d2, x_col, y_col, dataset, color, metadata_type,
                       preprocess_1, preprocess_2, label_1, label_2):
    """
    Create 1x2 bibiplot
    :param data_1: raw dataset 1 (dataframe)
    :param data_2: raw dataset 2 (dataframe)
    :param d1: projection of dataset 1 into latent space
    :param d2: projection of dataset 2 into latent space
    :param x_col: latent space component used as the x plotting dimension
    :param y_col: latent space component used as the y plotting dimension
    :param dataset: which dataset selected by user (visual/motor/upload)
    :param color: vector identifying groups for coloring plots
    :param metadata_type: name of metadata column to use
    :param preprocess_1: user-selected preprocessing method for dataset 1
    :param preprocess_2: user-selected preprocessing method for dataset 2
    :param label_1: label for dataset 1
    :param label_2: label for dataset 2
    :return:
    """

    # special handling for mouse datasets
    if dataset in ('motor', 'visual'):
        # from bibiplot_motor.ipynb
        geneExpr = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp.csv', index_col=0)
        geneExpr = pd.DataFrame(geneExpr, dtype='int')
        count = scipy.sparse.csr_matrix(geneExpr)  #.T  # transpose to put cells in rows, genes in columns
        seqDepth = np.sum(count, axis=1)  # sum gene enrichments across columns
        seqDepth = np.array(seqDepth)  # convert from matrix to vector
        X = np.log10(count / seqDepth * np.median(seqDepth) + 1)  # normalize  (not clear that multiplying by the median adds much value)
        X = np.array(X)
        X = X - X.mean(axis=0)
        X = X / np.std(X, axis=0)

        labels_X = geneExpr.columns.tolist()  # gene names

        ephysY = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature.csv', index_col=0)
        # if type(ephysY.iloc[0, 0]) == str:
        #     # drop the first column if it contains strings (i.e., presumably cell names)
        #     ephysY.drop(columns=ephysY.columns[0], inplace=True)
        Y = ephysY.to_numpy() - np.mean(ephysY.to_numpy(), axis=0)
        Y = Y / np.std(Y, axis=0)

        labels_Y = ephysY.columns.tolist()

    else:  # User-uploaded data
        labels_X = data_1.columns.tolist()
        labels_Y = data_2.columns.tolist()

        # preprocess using user-selected method
        X = preprocess(data_1, preprocess_1)
        Y = preprocess(data_2, preprocess_2)

    Zx = d1[:, (x_col, y_col)]
    Zx = Zx / np.std(Zx, axis=0)

    Zy = d2[:, (x_col, y_col)]
    Zy = Zy / np.std(Zy, axis=0)

    Rho_x = np.corrcoef(np.concatenate((Zx, X), axis=1), rowvar=False)[2:, :2]
    Rho_y = np.corrcoef(np.concatenate((Zy, Y), axis=1), rowvar=False)[2:, :2]

    titles = (label_1, label_2)

    # Set font size for titles without affecting other annotations
    titles = [f'<span style="font-size: {plot_title_font_size}px;">{t}</span>' for t in titles]

    fig = make_subplots(rows=2, cols=2,
                        row_heights=[0, 1],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.02,
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

    if color is None:
        # No color vector, so use a single group, single color, with no legend
        colors = 'black'
        use_categorical_colors = False
        show_legend = False
    else:

        categories = list(set(color))
        if len(categories) <= 10:  # arbitrary limit, but corresponds to length of color vector
            # category plot, show legend
            categories = sorted(categories)
            color_dict = {category: px.colors.qualitative.Plotly[i] for i, category in enumerate(categories)}
            use_categorical_colors = True
            show_legend = True
        else:
            # too many categories for legend.
            use_categorical_colors = False
            if color.dtype in ('string', 'object'):
                # plot black, no legend
                colors = 'black'
                show_legend = False
            else:
                # presumably numeric
                # continuous plot, show legend
                categories = sorted(categories)
                colors = color
                show_legend = True

    # Scatter plots
    for r in [2]:
        if use_categorical_colors:
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
        else:
            # Create a single plot, using a color vector instead of a single color
            c = 1
            fig.add_trace(go.Scatter(x=d1[:, 0], y=d1[:, 1], mode='markers',
                                     marker={'size': 2, 'color': colors, 'colorscale': 'Jet'},
                                     xaxis=f'x{plot_id(r, c)}',
                                     yaxis=f'y{plot_id(r, c)}',
                                     showlegend=False,
                                     hoverinfo="none"
                                     ),
                          row=r, col=c)
            c = 2
            fig.add_trace(go.Scatter(x=d2[:, 0], y=d2[:, 1], mode='markers',
                                     marker={'size': 2, 'color': colors, 'colorscale': 'Jet', 'showscale': show_legend},
                                     xaxis=f'x{plot_id(r, c)}',
                                     yaxis=f'y{plot_id(r, c)}',
                                     showlegend=False, #show_legend,
                                     hoverinfo="none"
                                     ),
                          row=r, col=c)

        # Circles
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
                                  row=r, col=c + 1,
                                  x0=0, y0=0,
                                  x1=Rho[i, 0], y1=Rho[i, 1],
                                  line={'color': 'black', 'width': 1})

                    if Rho[i, 0] < -0.2:
                        annotation_xanchor = 'left'
                    elif Rho[i, 0] > 0.2:
                        annotation_xanchor = 'right'
                    else:
                        annotation_xanchor = 'center'
                    fig.add_annotation(x=Rho[i, 0],
                                       y=Rho[i, 1],
                                       yshift=label_y_offset if Rho[i, 1] > -0.1 else -label_y_offset,
                                       xanchor=annotation_xanchor,
                                       text=labels[i],
                                       showarrow=False,
                                       row=r, col=c + 1)

            # Add labels to lines
            if False:
                fig.add_trace(go.Scatter(
                    x=Rho[sig_idx, 0], y=Rho[sig_idx, 1],
                    text=np.array(labels)[sig_idx],
                    mode='text',
                    showlegend=False,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=r, col=c + 1)

        # Axis labels, etc
        c = 1
        fig.update_yaxes(title_text=f'Component {y_col + 1}', row=r, col=c,
                         scaleanchor=f'x{plot_id(r, c)}', scaleratio=1,
                         showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)
        c = 2
        fig.update_yaxes(scaleanchor=f'x{plot_id(r, c)}', scaleratio=1, row=r, col=c,
                         showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)

        for c in [1, 2]:
            fig.update_xaxes(title_text=f'Component {x_col + 1}', row=r, col=c,
                             showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)
            fig.update_xaxes(row=r, col=c,
                             showticklabels=False, showgrid=False, showline=False, visible=True, zeroline=False)

    # This appears to link x and y axes so both plots have same level of pan/zoom
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig.update_layout(title_text='Top Feature Correlation with Latent Space',
                      plot_bgcolor='white',
                      font_size=plot_font_size,
                      title_font_size=plot_title_font_size)
    fig.update_annotations(font_size=plot_font_size)

    if show_legend:
        fig.update_layout(legend={'itemsizing': 'constant', 'title': color_types.get(color.name, color.name)})

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
