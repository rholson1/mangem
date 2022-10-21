import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import scale


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