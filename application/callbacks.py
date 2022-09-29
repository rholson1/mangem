from dash import Dash, html, dcc, Output, Input, State, MATCH, ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

from plots import create_bibiplot, create_alignment_plot
from operations.alignment import nonlinear_manifold_alignment
from operations.maninetcluster.util import Timer

import time

color_types = {'ttype': 't-type', 'gmm_cluster': 'Cluster'}


def register_callbacks(app, cache):
    @app.callback(
        Output('upload-container', 'className'),
        Output('store-data_selected', 'data'),
        Input('data-selector', 'value'),
    )
    def select_data(dataset):
        """Handle changes to the data selection dropdown
        """
        if dataset == 'upload':
            className = ''
        else:
            className = 'hidden'
        return className, str(time.time())


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
        Output('graph-combined', 'style'),
        Input('session_id', 'data'),
        Input(component_id='plot-type', component_property='value'),
        Input(component_id='color-type', component_property='value'),
        Input('store-aligned', 'data'),
        State('data-selector', 'value'),
        prevent_initial_call=True
    )
    def update_plot(session_id, plot_type, color_type, last_aligned, dataset):
        """Display visualization based on available data and selected visualization options
        """
        # read aligned data files.
        if dataset not in ('visual', 'motor'):
            return go.Figure()

        efeatures_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_NMA.csv')
        efeatures_NMA['gmm_cluster'] = efeatures_NMA['gmm_cluster'].astype('string')
        Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

        geneExp_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_NMA.csv')
        geneExp_NMA['gmm_cluster'] = geneExp_NMA['gmm_cluster'].astype('string')
        Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

        aligned = 'true'  # temporary override

        if aligned == 'true':
            # loading aligned data from cache
            Xe = np.array(pd.read_json(cache.get(f'{session_id}-aligned_1')))
            Xg = np.array(pd.read_json(cache.get(f'{session_id}-aligned_2')))
        else:
            return go.Figure(), {}

        # Xe = scale(Xe)
        # Xg = scale(Xg)

        style = {}  # default
        if plot_type == 'alignment':
            fig = create_alignment_plot(Xe, Xg, dataset)
            fig.update_layout()
            style = {'height': '600px', 'width': '1000px'}
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
            style = {'height': '600px', 'width': '1000px'}

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
            style = {'height': '600px', 'width': '1000px'}

        elif plot_type == 'bibiplot':
            fig = create_bibiplot(Xg, Xe)
            style = {'height': '600px', 'width': '600px'}
        else:
            # No plot type specified - show a blank plot
            fig = go.Figure().update_layout()

        return fig, style

    @app.callback(
        Output(component_id='loading-output-1', component_property='children'),
        Output('store-aligned', 'data'),
        Input('session_id', 'data'),
        Input(component_id='btn-align', component_property='n_clicks'),
        Input(component_id='data-selector', component_property='value'),
        State(component_id='eig-method', component_property='value'),
        State(component_id='eig-count', component_property='value'),

        prevent_initial_call=True
    )
    def align_datasets(session_id, value, dataset, eig_method, eig_count):
        # perform dataset alignment

        if dataset in ('motor', 'visual'):
            # use precomputed mouse cortex data
            pass
        else:
            # use uploaded data, if any
            return 'select mouse data for now'
            pass


        efeatures_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_filtered.csv')
        # Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

        # drop the first column if it contains strings
        if type(efeatures_NMA.iloc[1, 0]) == str:
            Xe = np.array(efeatures_NMA.iloc[:, 1:], dtype=float)
        else:
            Xe = np.array(efeatures_NMA, dtype=float)


        # from https://github.com/daifengwanglab/scMNC/blob/main/mouse_motor_cortex/code/dimreduction_clustering_motor.Rmd
        # it looks like columns of Xe need to be centered and scaled: subtract mean and divide by std dev.
        Xe = scale(Xe)
        geneExp_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_filtered.csv', header=0).T

        # geneExp_NMA['gmm_cluster'] = geneExp_NMA['gmm_cluster'].astype('string')
        # Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))
        Xg = np.array(geneExp_NMA[1:], dtype=float)  # skip gene row
        Xg = np.log2(Xg + 1)

        # parameters in R:
        # class = t_type
        # k_NN = 2
        # k_medoid = 5

        #with Timer('nonlinear manifold alignment (ortho)'):
        proj, _ = nonlinear_manifold_alignment(Xe, Xg, 20, eig_method=eig_method, eig_count=int(eig_count))

        # store aligned datasets for later retrieval
        aligned_1, aligned_2 = proj
        cache.set(f'{session_id}-aligned_1', pd.DataFrame(aligned_1).to_json())
        cache.set(f'{session_id}-aligned_2', pd.DataFrame(aligned_2).to_json())
        return '', str(time.time())

    @app.callback(
        Output(component_id='loading-output-2', component_property='children'),
        Input(component_id='btn-cluster', component_property='n_clicks'),
        prevent_initial_call=True
    )
    def identify_clusters(value):
        # identify clusters
        time.sleep(2)
        return 'Clusters identified'


    # @app.callback(
    #     Output('store-aligned', 'data'),
    #     Output('alignment-state', 'children'),
    #     Input('btn-align', 'n_clicks'),
    #     Input('data-selector', 'value'),
    # )
    # def set_alignment_state(button, dataset):
    #     """ If the data are aligned (via clicking on the alignment button) then set alignment state to true.
    #     If anything else changes that could invalidate the alignment, set alignment state to false.
    #
    #     This will likely need to change if/when alignment is automatically triggered.
    #     """
    #     if ctx.triggered_id == 'btn-align':
    #         return 'true', 'Status: Aligned'
    #     else:
    #         return 'false', 'Status: Not Aligned'




