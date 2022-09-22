from dash import Dash, html, dcc, Output, Input, State, MATCH
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

from plots import create_bibiplot
from operations.alignment import nonlinear_manifold_alignment
from operations.maninetcluster.util import Timer

import time

color_types = {'ttype': 't-type', 'gmm_cluster': 'Cluster'}


def register_callbacks(app, cache):
    @app.callback(
        Output({'type': 'dynamic-output', 'index': MATCH}, 'children'),
        Input({'type': 'dynamic-upload', 'index': MATCH}, 'contents'),
        State({'type': 'dynamic-upload', 'index': MATCH}, 'filename'),
        State({'type': 'dynamic-upload', 'index': MATCH}, 'last_modified')
    )
    def handle_upload(contents, filename, last_modified):
        return filename

    @app.callback(
        Output(component_id='graph-combined', component_property='figure'),
        Input('session_id', 'data'),
        Input(component_id='mouse-selector', component_property='value'),
        Input(component_id='plot-type', component_property='value'),
        Input(component_id='color-type', component_property='value'),
        prevent_initial_call=True
    )
    def update_alignment_plot(session_id, dataset, plot_type, color_type):
        # read aligned data files.
        if not dataset:
            return go.Figure()

        efeatures_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_NMA.csv')
        efeatures_NMA['gmm_cluster'] = efeatures_NMA['gmm_cluster'].astype('string')
        Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

        geneExp_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_NMA.csv')
        geneExp_NMA['gmm_cluster'] = geneExp_NMA['gmm_cluster'].astype('string')
        Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

        # loading aligned data from cache
        Xe = np.array(pd.read_json(cache.get(f'{session_id}-aligned_1')))
        Xg = np.array(pd.read_json(cache.get(f'{session_id}-aligned_2')))


        if plot_type == 'alignment':
            # plot both sets of data (first three dimensions)
            fig_e = px.scatter_3d(Xe, x=0, y=1, z=2).update_traces(name='Electrophys',
                                                                   marker={'color': 'red', 'size': 0.75},
                                                                   showlegend=True)
            fig_g = px.scatter_3d(Xg, x=0, y=1, z=2).update_traces(name='Gene Expression',
                                                                   marker={'color': 'blue', 'size': 0.75},
                                                                   showlegend=True)
            fig = go.Figure(data=fig_e.data + fig_g.data)

            fig.update_layout(
                title=f'Multimodal Alignment in Latent Space: Mouse {dataset.title()} Cortex',
                showlegend=True,
                scene={
                    'xaxis_title': '',
                    'yaxis_title': '',
                    'zaxis_title': '',
                },
                legend={'itemsizing': 'constant'}
            )
        elif plot_type == 'separate2':
            # Create two 2-D plots side by side for the two modalities.
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Electrophys', 'Gene Expression'))

            if color_type:
                fig_e = px.scatter(efeatures_NMA, x='Val0', y='Val1', color=color_type)
                fig_g = px.scatter(geneExp_NMA, x='Val0', y='Val1', color=color_type)

                # Suppress legend for one of the plots to avoid double entries
                for d in fig_e.data:
                    d['showlegend'] = False

                # sort fig_*.data to get a sorted legend
                fig.add_traces(sorted(fig_e.data, key=itemgetter('legendgroup')), rows=1, cols=1)
                fig.add_traces(sorted(fig_g.data, key=itemgetter('legendgroup')), rows=1, cols=2)

                # Set the legend title
                fig.update_layout(legend_title=color_types[color_type], legend={'itemsizing': 'constant'})

            else:
                # Uncolored plots
                fig_e = px.scatter(efeatures_NMA, x='Val0', y='Val1')
                fig_g = px.scatter(geneExp_NMA, x='Val0', y='Val1')

                fig.add_trace(fig_e.data[0], row=1, col=1)
                fig.add_trace(fig_g.data[0], row=1, col=2)

        elif plot_type == 'separate3':
            # Create two 2-D plots side by side for the two modalities.
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Electrophys', 'Gene Expression'),
                                specs=[[{'type': 'scene'}, {'type': 'scene'}]])

            if color_type:
                fig_e = px.scatter_3d(efeatures_NMA, x='Val0', y='Val1', z='Val2', color=color_type).update_traces(
                    marker={'size': 0.75})
                fig_g = px.scatter_3d(geneExp_NMA, x='Val0', y='Val1', z='Val2', color=color_type).update_traces(
                    marker={'size': 0.75})

                # Suppress legend for one of the plots to avoid double entries
                for d in fig_e.data:
                    d['showlegend'] = False

                # sort fig_*.data to get a sorted legend
                fig.add_traces(sorted(fig_e.data, key=itemgetter('legendgroup')), rows=1, cols=1)
                fig.add_traces(sorted(fig_g.data, key=itemgetter('legendgroup')), rows=1, cols=2)

                # Set the legend title
                fig.update_layout(legend_title=color_types[color_type], legend={'itemsizing': 'constant'})

            else:
                # Uncolored plots
                fig_e = px.scatter_3d(efeatures_NMA, x='Val0', y='Val1', z='Val2')
                fig_g = px.scatter_3d(geneExp_NMA, x='Val0', y='Val1', z='Val2')

                fig.add_trace(fig_e.data[0], row=1, col=1)
                fig.add_trace(fig_g.data[0], row=1, col=2)
        elif plot_type == 'bibiplot':
            return create_bibiplot()
        else:
            # No plot type specified - show a blank plot
            fig = go.Figure().update_layout()

        return fig

    @app.callback(
        Output(component_id='loading-output-1', component_property='children'),
        Input('session_id', 'data'),
        Input(component_id='btn-align', component_property='n_clicks'),
        State(component_id='eig-method', component_property='value'),
        State(component_id='mouse-selector', component_property='value'),
        prevent_initial_call=True
    )
    def align_datasets(session_id, value, eig_method, mouse_set):
        # perform dataset alignment

        day_ortho = np.genfromtxt("data/maninetcluster/dayOrthoExpr.csv", delimiter=',')[1:, 1:]
        night_ortho = np.genfromtxt("data/maninetcluster/nightOrthoExpr.csv", delimiter=',')[1:, 1:]
        num_dims = 2

        dataset = 'motor'
        efeatures_NMA = pd.read_csv(f'data/mouse_{mouse_set}_cortex/efeature_filtered.csv') #, dtype=float)
        # Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

        # drop the first column if it contains strings
        if type(efeatures_NMA.iloc[1, 0]) == str:
            Xe = np.array(efeatures_NMA.iloc[:, 1:], dtype=float)
        else:
            Xe = np.array(efeatures_NMA, dtype=float)


        # from https://github.com/daifengwanglab/scMNC/blob/main/mouse_motor_cortex/code/dimreduction_clustering_motor.Rmd
        # it looks like columns of Xe need to be centered and scaled: subtract mean and divide by std dev.
        Xe = scale(Xe)
        geneExp_NMA = pd.read_csv(f'data/mouse_{mouse_set}_cortex/geneExp_filtered.csv', header=0).T

        # geneExp_NMA['gmm_cluster'] = geneExp_NMA['gmm_cluster'].astype('string')
        # Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))
        Xg = np.array(geneExp_NMA[1:], dtype=float)  # skip gene row
        Xg = np.log2(Xg + 1)

        # parameters in R:
        # class = t_type
        # k_NN = 2
        # k_medoid = 5

        #with Timer('nonlinear manifold alignment (ortho)'):
        proj, _ = nonlinear_manifold_alignment(Xe, Xg, 20, eig_method=eig_method)

        # store aligned datasets for later retrieval
        aligned_1, aligned_2 = proj
        cache.set(f'{session_id}-aligned_1', pd.DataFrame(aligned_1).to_json())
        cache.set(f'{session_id}-aligned_2', pd.DataFrame(aligned_2).to_json())
        return 'Dataset aligned'

    @app.callback(
        Output(component_id='loading-output-2', component_property='children'),
        Input(component_id='btn-cluster', component_property='n_clicks'),
        prevent_initial_call=True
    )
    def identify_clusters(value):
        # identify clusters
        time.sleep(2)
        return 'Clusters identified'
