import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import scale
from operations.preprocessing import preprocess
from app_main.constants import font_size


def create_heatmap2(session_id, dataset, data_1, data_2, preprocess_1, preprocess_2, cell_cluster, num_clusters,
                    label_1, label_2, num_enriched, size_key='default'):
    """ Create heatmap showing differentially-expressed genes by cluster
    """

    plot_title_font_size = font_size[size_key]['plot_title_font_size']

    top_enriched = {}

    fig = make_subplots(rows=2, cols=2,
                        row_heights=[0.05, 0.95],
                        vertical_spacing=0.02,
                        horizontal_spacing=0.2,
                        column_titles=[label_1, label_2],
                        shared_xaxes=True)

    for col, data, preprocess_method in zip([1, 2], [data_1, data_2], [preprocess_1, preprocess_2]):
        # preprocess using user-selected method
        data = preprocess(data, preprocess_method)
        # then standardize each column
        data = preprocess(data, 'scale')

        adata = AnnData(data, dtype='float')
        adata.obs['cluster'] = pd.Categorical(cell_cluster)

        sc.tl.rank_genes_groups(adata, 'cluster', method='wilcoxon')

        # Get the names of the top num_enriched genes for each cluster
        top10genes = [adata.uns['rank_genes_groups']['names'][r][c] for c in range(num_clusters) for r in range(num_enriched)]

        # Store ranked feature data
        top_enriched[col] = pd.DataFrame(data=adata.uns['rank_genes_groups']['names'][:num_enriched])

        # find order to sort cells by cluster
        cell_order = np.argsort(cell_cluster)
        # sort cluster vector by cluster order (for the top bar)
        cluster = cell_cluster[cell_order]
        display_data = data[top10genes].T.iloc[:, cell_order]

        x = [str(s) for s in display_data.columns]  # Convert to strings so numeric column names not used as indices

        fig.add_trace(go.Heatmap(z=cluster, x=x, y=np.ones(len(cluster)),
                                 colorscale='spectral', showscale=False,
                                 hovertemplate='Cluster %{z}<extra></extra>'),
                      row=1, col=col)
        fig.add_trace(go.Heatmap(z=display_data,
                                 x=x, #display_data.columns.values,
                                 y=display_data.index,
                                 colorscale='gnbu',
                                 zmax=2,
                                 zmin=-1,
                                 showscale=False,
                                 hovertemplate='Cell: %{x}<br>Feature: %{y}<extra></extra>'
                                 ),
                      row=2, col=col)

        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)

        fig.update_xaxes(showticklabels=False, row=2, col=col)
        fig.update_yaxes(autorange='reversed', row=2, col=col)

    fig.update_layout(title_text=f'Cross-modal Cluster Feature Expression',
                      #font_size=plot_font_size,
                      title_font_size=plot_title_font_size)
    fig.update_annotations(font_size=plot_title_font_size)

    return fig, top_enriched


def create_heatmap_general(data_1, data_2, clusters,
                           preprocess_1, preprocess_2):
    """ Create heatmap showing differentially-expressed genes by cluster

    For each dataset, require the raw dataset (for computing gene/feature expression)
    as well as cluster information
    """

    labels_X = data_1.columns.tolist()
    labels_Y = data_2.columns.tolist()

    # preprocess using user-selected method
    X = preprocess(data_1, preprocess_1)
    Y = preprocess(data_2, preprocess_2)

    # then standardize each column
    if preprocess_1 != 'scale':
        X = preprocess(X, 'scale')
    if preprocess_2 != 'scale':
        Y = preprocess(Y, 'scale')

    fig = make_subplots(rows=2, cols=2, row_heights=[0.1, 0.9], vertical_spacing=0.02)

    cell_order = np.argsort(clusters)
    cluster = clusters[cell_order]

    for col, D in enumerate([X, Y], start=1):
        adata = AnnData(D, dtype='float')
        adata.obs['cluster'] = pd.Categorical(clusters)
        sc.tl.rank_genes_groups(adata, 'cluster', method='wilcoxon')
        top10genes = [adata.uns['rank_genes_groups']['names'][r][c] for c in range(5) for r in range(10)]

        data = D[top10genes].T.iloc[:, cell_order]

        fig.add_trace(go.Heatmap(z=cluster, x=data.columns.values, y=np.ones(len(cluster)),
                                 colorscale='spectral', showscale=False,
                                 hovertemplate='Cluster %{z}<extra></extra>'),
                      row=1, col=col)
        fig.add_trace(go.Heatmap(z=data,
                                 x=data.columns.values,
                                 y=list(data.index),
                                 hovertemplate='%{x}<br>%{y}<extra></extra>'
                                 ),
                      row=2, col=col)


        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)

        fig.update_xaxes(showticklabels=False, row=2, col=col)
        fig.update_yaxes(autorange='reversed', row=2, col=col)


    fig.update_xaxes(matches='x')  # this changes the behavior after bibiplot

    fig.update_layout(title_text=f'Feature Enrichment by Cluster')

    return fig
