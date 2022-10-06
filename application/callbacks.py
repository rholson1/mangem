from dash import Dash, html, dcc, Output, Input, State, MATCH, ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

from plots import create_bibiplot1x2, create_alignment_plot, scatter2d, scatter3d, plot_alignment_error, plot_alignment
from operations.alignment import nonlinear_manifold_alignment
from operations.clustering import cluster_gmm
from operations.preprocessing import preprocess
from operations.maninetcluster.util import Timer

import time

color_types = {'ttype': 't-type', 'cluster': 'Cluster'}


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
        Input('component_x', 'value'),
        Input('component_y', 'value'),
        Input('component_z', 'value'),
        Input('store-aligned', 'data'),
        State('data-selector', 'value'),
        prevent_initial_call=True
    )
    def update_plot(session_id, plot_type, color_type, x, y, z, last_aligned, dataset):
        """Display visualization based on available data and selected visualization options
        """
        # read aligned data files.

        # Convert component numbers to integers and make 0-based instead of 1-based
        x = int(x) - 1
        y = int(y) - 1
        z = int(z) - 1

        if dataset not in ('visual', 'motor'):
            return go.Figure()

        if False:
            efeatures_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_NMA.csv')
            efeatures_NMA['gmm_cluster'] = efeatures_NMA['gmm_cluster'].astype('string')
            Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

            geneExp_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_NMA.csv')
            geneExp_NMA['gmm_cluster'] = geneExp_NMA['gmm_cluster'].astype('string')
            Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

        aligned = 'true'  # temporary override

        if aligned == 'true':
            # loading aligned data from cache
            efeatures_NMA = pd.read_json(cache.get(f'{session_id}-aligned_1'))
            efeatures_NMA['cluster'] = efeatures_NMA['cluster'].astype('string')
            Xe = np.array(efeatures_NMA.drop(columns=['cluster', 'ttype']))

            geneExp_NMA = pd.read_json(cache.get(f'{session_id}-aligned_2'))
            geneExp_NMA['cluster'] = geneExp_NMA['cluster'].astype('string')
            Xg = np.array(geneExp_NMA.drop(columns=['cluster', 'ttype']))

            #Xe = np.array(pd.read_json(cache.get(f'{session_id}-aligned_1')))
            #Xg = np.array(pd.read_json(cache.get(f'{session_id}-aligned_2')))
        else:
            return go.Figure(), {}


        style = {}  # default
        if plot_type == 'alignment':
            #fig = create_alignment_plot(Xe, Xg, dataset, x, y, z)
            fig = plot_alignment(Xe, Xg, dataset, x, y, z)
            style = {'height': '600px', 'width': '1000px'}

        elif plot_type == 'alignment-error':
            fig = plot_alignment_error(Xe, Xg, dataset)
            style = {'height': '600px', 'width': '1000px'}

        elif plot_type == 'separate2':
            fig = scatter2d(efeatures_NMA, geneExp_NMA, x, y, color_type)
            style = {'height': '600px', 'width': '1000px'}

        elif plot_type == 'separate3':
            fig = scatter3d(efeatures_NMA, geneExp_NMA, x, y, z, color_type)
            style = {'height': '600px', 'width': '1000px'}

        elif plot_type == 'bibiplot':
            fig = create_bibiplot1x2(Xg, Xe, x, y)
            style = {'height': '600px', 'width': '1000px'}

        else:
            # No plot type specified - show a blank plot
            fig = go.Figure().update_layout()
            #style = {'height': '0', 'width': '0'}

        return fig, style

    @app.callback(
        Output(component_id='loading-output-1', component_property='children'),
        Output('store-aligned', 'data'),
        Input('session_id', 'data'),
        Input('btn-align', 'n_clicks'),
        Input('btn-cluster', 'n_clicks'),
        Input(component_id='data-selector', component_property='value'),
        State('preprocess_1', 'value'),
        State('preprocess_2', 'value'),
        State(component_id='eig-method', component_property='value'),
        State(component_id='eig-count', component_property='value'),

        prevent_initial_call=True
    )
    def align_and_cluster_datasets(session_id, align_clicks, cluster_clicks, dataset,
                                   preprocess_1, preprocess_2,
                                   eig_method, eig_count):

        # add logic for selectively processing only what has to be processed
        # (e.g., if clustering button is clicked, don't re-run alignment (which is expensive!)
        if ctx.triggered_id == 'btn-cluster':
            # load aligned data from cache
            df_1 = pd.read_json(cache.get(f'{session_id}-aligned_1'))
            df_2 = pd.read_json(cache.get(f'{session_id}-aligned_2'))
            ttype = df_1['ttype']
            aligned_1 = np.array(df_1.drop(columns=['ttype', 'cluster']))
            aligned_2 = np.array(df_2.drop(columns=['ttype', 'cluster']))
        else:

            # perform dataset alignment and clustering

            if dataset in ('motor', 'visual'):
                # use precomputed mouse cortex data
                pass
            else:
                # use uploaded data, if any
                return 'select mouse data for now', str(time.time())

            # Load mouse data and clean based on prior knowledge of structure.
            df_1 = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_filtered.csv')
            # drop the first column if it contains strings
            if type(df_1.iloc[1, 0]) == str:
                Xe = np.array(df_1.iloc[:, 1:], dtype=float)
            else:
                Xe = np.array(df_1, dtype=float)

            df_2 = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_filtered.csv', header=0).T
            Xg = np.array(df_2[1:], dtype=float)  # skip gene row

            # Load metadata files containing ttype column
            ttype = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_NMA_col.csv')['ttype']





            # Apply selected preprocessing to raw datasets
            Xe = preprocess(Xe, preprocess_1)
            Xg = preprocess(Xg, preprocess_2)

            # Align datasets
            proj, _ = nonlinear_manifold_alignment(Xe, Xg, 20, eig_method=eig_method, eig_count=int(eig_count))
            aligned_1, aligned_2 = proj

        # Identify clusters (generalize later with clustering parameters, alternate methods)
        clusters = cluster_gmm(aligned_1, aligned_2)

        # Append cluster, t_type info to aligned data in a Pandas dataframe
        aligned_1 = pd.DataFrame(aligned_1)
        aligned_1['cluster'] = clusters
        aligned_1['ttype'] = ttype
        aligned_2 = pd.DataFrame(aligned_2)
        aligned_2['cluster'] = clusters
        aligned_2['ttype'] = ttype

        # Store aligned and annotated data in cache
        cache.set(f'{session_id}-aligned_1', aligned_1.to_json())
        cache.set(f'{session_id}-aligned_2', aligned_2.to_json())
        return '', str(time.time())


    @app.callback(
        Output('component_x', 'max'),
        Output('component_y', 'max'),
        Input('eig-count', 'value')
    )
    def set_max_component(eig_count):
        return eig_count, eig_count

    # @app.callback(
    #     Output(component_id='loading-output-2', component_property='children'),
    #     Input(component_id='btn-cluster', component_property='n_clicks'),
    #     prevent_initial_call=True
    # )
    # def identify_clusters(value):
    #     # identify clusters
    #     time.sleep(2)
    #
    #     # GMM
    #
    #
    #     return 'Clusters identified'


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




