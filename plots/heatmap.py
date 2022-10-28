import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import scale
from operations.preprocessing import preprocess


def create_heatmap(dataset, geneExp_NMA):
    """ Create heatmap showing differentially-expressed genes by cluster

    Need to figure out how to generalize to user-uploaded input, but implement for mouse datasets first

    The existing code uses the R package Seurat:



    #different expressed genes
    diffg <- CreateSeuratObject(counts =log10(gdata+1), min.cells = 3, min.features = 200)
    all.genes <- rownames(diffg)
    diffg <- ScaleData(diffg, features = all.genes)
    Idents(diffg) = factor(paste0("Cluster ",gmm_cluster))
    diffg.markers <- FindAllMarkers(diffg, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
    top10 <- diffg.markers  %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)
    #Figure 5A
    DoHeatmap(diffg, features = top10$gene,group.colors = brewer.pal(6,"Dark2")[1:5],angle = 30,size = 5)


    Seurat uses the Wilcox Rank Sum test in FindAllMarkers.  This is also implemented in scipy.stats.ranksums


    scanpy might be a python-based alternative to Seurat
    sc.pl.heatmap is very similar to DoHeatmap (though it's matplotlib based) -- store as .png, and then display?
    scanpy.tl.rank_genes_groups (or filter_rank_genes_groups) may work



    """

    geneExp = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_filtered.csv', index_col=0)

    labels_X = geneExp.index.array  # gene names

    # data should be log transformed, and transpose so rows correspond to cells (and columns, genes)
    geneExp = np.log10(geneExp.T + 1)

    # standardize data (subtract mean and divide by StdDev) for each gene.
    for gene in labels_X:
        geneExp[gene] = scale(geneExp[gene])


    adata = AnnData(geneExp, dtype='float')
    adata.obs['cluster'] = pd.Categorical(geneExp_NMA['cluster'])

    sc.tl.rank_genes_groups(adata, 'cluster', method='wilcoxon')
    # Get the names of the top 10 genes for each cluster
    #top10genes = adata.uns['rank_genes_groups']['names'][:10]


    top10genes = [adata.uns['rank_genes_groups']['names'][r][c] for c in range(5) for r in range(10)]

    cell_order = np.argsort(geneExp_NMA['cluster'])
    cluster = geneExp_NMA['cluster'][cell_order]
    data = geneExp[top10genes]
    data1 = data.T.iloc[:, cell_order]
    # this works
    #fig = px.imshow(data1, aspect='auto')


    fig = make_subplots(rows=2, cols=1, row_heights=[0.1, 0.9], vertical_spacing=0.02)

    fig.add_trace(go.Heatmap(z=cluster, x=data1.columns.values, y=np.ones(len(cluster)),
                             colorscale='spectral', showscale=False,
                             hovertemplate='Cluster %{z}<extra></extra>'),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=data1,
                             x=data1.columns.values,
                             y=list(data1.index),
                             hovertemplate='Cell: %{x}<br>Gene: %{y}<extra></extra>'
                             ),
                  row=2, col=1)

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(autorange='reversed', row=2, col=1)

    fig.update_xaxes(matches='x')  # this changes the behavior after bibiplot

    fig.update_layout(title_text=f'Gene Enrichment by Cluster: Mouse {dataset.title()} Cortex')

    # fig = go.Figure(data=go.Heatmap(z=data1,
    #                                 x=data1.columns.values,
    #                                 y=list(data1.index)))
    # fig.update_yaxes(autorange='reversed')


    #fig.update_xaxes(side='top')
    #fig = go.Figure().update_layout()
    return fig


def create_heatmap2(dataset, data_1, data_2, preprocess_1, preprocess_2, cell_cluster, num_clusters,
                    label_1, label_2):
    """ Create heatmap showing differentially-expressed genes by cluster
    """

    #geneExp = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_filtered.csv', index_col=0)



    # # data should be log transformed, and transpose so rows correspond to cells (and columns, genes)
    # geneExp = np.log10(geneExp.T + 1)
    #
    # # standardize data (subtract mean and divide by StdDev) for each gene.
    # for gene in labels_X:
    #     geneExp[gene] = scale(geneExp[gene])

    fig = make_subplots(rows=2, cols=2,
                        row_heights=[0.1, 0.9],
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

        # Get the names of the top 10 genes for each cluster
        top10genes = [adata.uns['rank_genes_groups']['names'][r][c] for c in range(num_clusters) for r in range(10)]

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
                                 showscale=False,
                                 hovertemplate='Cell: %{x}<br>Gene: %{y}<extra></extra>'
                                 ),
                      row=2, col=col)

        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)

        fig.update_xaxes(showticklabels=False, row=2, col=col)
        fig.update_yaxes(autorange='reversed', row=2, col=col)

    #fig.update_xaxes(matches='x')  # this changes the behavior after bibiplot

    fig.update_layout(title_text=f'Feature Enrichment by Cluster')

    return fig



#def create_heatmap_general(dataset, geneExp_NMA):
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
