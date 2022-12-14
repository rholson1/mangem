from dash import Output, Input, State, MATCH, ALL, ctx, dcc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

from plots import *
from operations.alignment import alignment, UnexpectedAlignmentMethodException #nonlinear_manifold_alignment
from operations.clustering import cluster_gmm, cluster_kmeans, cluster_hierarchical
from operations.preprocessing import preprocess
from operations.maninetcluster.util import Timer

from app_main.settings import cell_limit
from app_main.utilities import safe_filenames, cache_key
from app_main.constants import UploadFileType, blank_layout

import io
import base64
import zipfile

import time


def register_callbacks(app, cache):
    @app.callback(
        Output('upload-container', 'className'),
        Output('store-data_selected', 'data'),
        Output('upload_1_label', 'value'),
        Output('upload_2_label', 'value'),
        Input('data-selector', 'value'),
    )
    def select_data(dataset):
        """Handle changes to the data selection dropdown
        """
        if dataset in ('motor', 'visual'):
            class_name = 'hidden'
            label_1 = 'Gene Expression'
            label_2 = 'Electrophys'
        elif dataset == 'upload':
            class_name = ''
            label_1 = ''
            label_2 = ''
        else:  # no selection
            class_name = 'hidden'
            label_1 = ''
            label_2 = ''

        return class_name, str(time.time()), label_1, label_2

    @app.callback(
        Output('store-label-1', 'data'),
        Output('store-label-2', 'data'),
        Input('upload_1_label', 'value'),
        Input('upload_2_label', 'value')
    )
    def update_label_store(label_1, label_2):
        return label_1 or 'Modality 1', label_2 or 'Modality 2'

    @app.callback(
        Output('preprocess-label-1', 'children'),
        Output('preprocess-label-2', 'children'),
        Output('explore-label-1', 'children'),
        Output('explore-label-2', 'children'),
        Input('store-label-1', 'data'),
        Input('store-label-2', 'data')
    )
    def update_preprocessing_labels(label_1, label_2):
        return label_1, label_2, label_1, label_2

    @app.callback(
        Output({'type': 'dynamic-output', 'index': MATCH}, 'children'),
        Output({'type': 'dynamic-upload-info', 'index': MATCH}, 'children'),
        Output({'type': 'loading-upload', 'index': MATCH}, 'children'),
        Output({'type': 'store-upload', 'index': MATCH}, 'data'),
        Input({'type': 'dynamic-upload', 'index': MATCH}, 'contents'),
        State({'type': 'dynamic-upload', 'index': MATCH}, 'filename'),
        State({'type': 'dynamic-upload', 'index': MATCH}, 'last_modified'),
        State({'type': 'dynamic-upload', 'index': MATCH}, 'id'),
        State('session_id', 'data'),
        prevent_initial_call=True
    )
    def handle_upload(contents, filename, last_modified, upload_id, session_id):
        """ Handle uploads of data files and metadata.  Return the filename (for display) and also
        store the file in the cache."""

        # Read data from file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=cell_limit)

        rows, columns = df.shape
        # determine the file type based on the widget used to upload the file
        file_type = UploadFileType(upload_id['index']).name

        # store data in cache
        key = cache_key(session_id, file_type)
        cache.set(key, df.to_json())

        return filename, f'Rows: {rows}, Columns: {columns}', '', str(time.time())

    @app.callback(
        Output('metadata-type-x', 'options'),
        Input({'type': 'store-upload', 'index': UploadFileType.METADATA}, 'data'),
        Input('data-selector', 'value'),
        State('session_id', 'data'),
        prevent_initial_call=True
    )
    def populate_metadata_x(metadata_uploaded, dataset, session_id):
        """Populate the metadata dropdown box for exploring the active dataset when the user either selects a mouse
        dataset or uploads metadata"""

        if ctx.triggered_id == 'data-selector':
            if dataset in ('motor', 'visual'):
                metadata_options = {'ttype': 'Transcriptomic Type (ttype)'}
            else:
                raise PreventUpdate
        else:
            # triggered by uploading a metadata file
            try:
                metadata_df = pd.read_json(cache.get(cache_key(session_id, UploadFileType.METADATA.name)))
                metadata_options = metadata_df.columns
            except ValueError:
                metadata_options = {}
        return metadata_options


    @app.callback(
        Output('metadata-value', 'options'),
        Input('metadata-type-x', 'value'),
        Input('data-selector', 'value'),
        State({'type': 'dynamic-output', 'index': ALL}, 'children'),
        State('session_id', 'data'),
        prevent_initial_call=True
    )
    def populate_metadata_value(metadata_type, dataset, upload_filenames, session_id):
        """Set the options for a dropdown as possible values of the selected metadata type"""

        if ctx.triggered_id == 'data-selector':
            # might as well load values from the metadata file
            if dataset in ('motor', 'visual'):
                ttype = pd.read_csv(f'data/mouse_{dataset}_cortex/metadata.csv')['ttype']
                return ttype.unique()
        else:
            # triggered by a change in metadata type
            if upload_filenames[2]:  # only try to read metadata from cache if a file has been uploaded
                metadata_df = pd.read_json(cache.get(cache_key(session_id, UploadFileType.METADATA.name)))
                metadata = metadata_df[metadata_type]
                return metadata.unique()
        return {}


    @app.callback(
        Output('explore-var1', 'options'),
        Output('explore-var2', 'options'),
        Input({'type': 'store-upload', 'index': UploadFileType.DATA_1}, 'data'),
        Input({'type': 'store-upload', 'index': UploadFileType.DATA_2}, 'data'),
        Input('data-selector', 'value'),
        State('session_id', 'data'),
        State('explore-var1', 'options'),
        State('explore-var2', 'options'),
        prevent_initial_call=True
    )
    def populate_explore_vars(d1_uploaded, d2_uploaded, dataset, session_id, options_1, options_2):
        """Populate the feature dropdowns in the explore data section of the data tab"""

        if ctx.triggered_id == 'data-selector':
            if dataset in ('motor', 'visual'):
                df_1 = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp.csv', index_col=0)
                df_2 = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature.csv', index_col=0)
                return sorted(df_1.columns), sorted(df_2.columns)
            else:
                raise PreventUpdate
        elif ctx.triggered_id == {'type': 'store-upload', 'index': UploadFileType.DATA_1}:
            df = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_1.name)))
            return sorted(df.columns), options_2
        elif ctx.triggered_id == {'type': 'store-upload', 'index': UploadFileType.DATA_2}:
            df = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_2.name)))
            return options_1, sorted(df.columns)
        else:
            raise PreventUpdate


    @app.callback(
        Output('explore-vars', 'value'),
        Input('use-explore-var1', 'n_clicks'),
        Input('use-explore-var2', 'n_clicks'),
        State('explore-var1', 'value'),
        State('explore-var2', 'value'),
        State('explore-vars', 'value'),
        prevent_initial_call=True
    )
    def select_explore_var(use_btn_1, use_btn_2, var_1, var_2, explore_vars):
        """ Copy a variable to the select box"""

        if explore_vars:
            xv = [s.strip() for s in explore_vars.split(',')]
        else:
            xv = []

        if ctx.triggered_id == 'use-explore-var1':
            if var_1:
                xv = xv[-1:] + [var_1]
            else:
                raise PreventUpdate
        elif ctx.triggered_id == 'use-explore-var2':
            if var_2:
                xv = xv[-1:] + [var_2]
            else:
                raise PreventUpdate

        return ', '.join(xv)


    @app.callback(
        Output('graph-x', 'figure'),
        Output('user-data-alert-x', 'children'),  # error message
        Output('user-data-alert-x', 'is_open'),
        Input('explore-vars', 'value'),
        Input('metadata-type-x', 'value'),
        Input('metadata-value', 'value'),
        Input('explore-preprocess', 'value'),
        Input('explore-log-axis', 'value'),
        State('data-selector', 'value'),
        State('preprocess_1', 'value'),
        State('preprocess_2', 'value'),
        State({'type': 'dynamic-output', 'index': ALL}, 'children'),
        State('session_id', 'data'),
        prevent_initial_call=True
    )
    def handle_explore_vars(explore_vars, metadata_type, metadata_value, apply_preprocess, log_axis,
                            dataset, preprocess_1, preprocess_2, upload_filenames, session_id):
        """

        :param explore_vars:
        :param metadata_type:
        :param apply_preprocess:
        :param log_axis:
        :param dataset:
        :param preprocess_1:
        :param preprocess_2:
        :param upload_filenames:
        :param session_id:
        :return:
        """

        if not (explore_vars and metadata_type):
            raise PreventUpdate

        error_message = ''
        fig = go.Figure(data={}, layout=blank_layout)

        vars = [s.strip().lower() for s in explore_vars.split(',')[:2]]

        if dataset in ('motor', 'visual'):
            df_1 = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp.csv', index_col=0)
            df_2 = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature.csv', index_col=0)
            metadata = pd.read_csv(f'data/mouse_{dataset}_cortex/metadata.csv', index_col=0)
        elif dataset == 'upload':
            try:
                df_1 = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_1.name)))
                df_2 = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_2.name)))
                metadata = pd.read_json(cache.get(cache_key(session_id, UploadFileType.METADATA.name)))
            except ValueError:
                error_message = 'Two data files must be uploaded before data can be explored.'
                return fig, error_message, bool(error_message)
        else:
            error_message = 'Data must be selected or uploaded before it can be explored.'
            return fig, error_message, bool(error_message)

        # convert columns to lower case
        df_1.columns = df_1.columns.str.lower()
        df_2.columns = df_2.columns.str.lower()
        df_1_cols = list(df_1.columns)
        df_2_cols = list(df_2.columns)
        possible_cols = df_1_cols + df_2_cols

        bad_vars = [v for v in vars if v.lower() not in possible_cols]
        if bad_vars:
            error_message = f'{bad_vars} not found in the selected data files.  Check for correct spelling.'
            return fig, error_message, bool(error_message)

        # Generate a plot
        plot_df = pd.DataFrame()

        apply_preprocess = apply_preprocess == ['preprocess']
        for v in vars:
            if v in df_1_cols:
                plot_df[v] = preprocess(df_1[v], preprocess_1) if apply_preprocess else df_1[v]
            elif v in df_2_cols:
                plot_df[v] = preprocess(df_2[v], preprocess_2) if apply_preprocess else df_2[v]
        plot_df[metadata_type] = metadata[metadata_type]

        if metadata_value:
            plot_df_filtered = plot_df.loc[plot_df[metadata_type] == metadata_value]
        else:
            plot_df_filtered = plot_df

        if len(vars) == 1:
            # Generate a box plot
            fig = px.box(plot_df_filtered, x=metadata_type, y=vars[0])
        elif len(vars) == 2:
            # Generate a scatter plot
            log_axis = log_axis or []  # handle log_axis == None
            fig = px.scatter(plot_df_filtered, x=vars[0], y=vars[1], color=metadata_type,
                             log_x='X' in log_axis, log_y='Y' in log_axis)
        else:
            raise Exception('Unexpected number of vars!  Bug!')

        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

        return fig, error_message, bool(error_message)


    @app.callback(
        Output(component_id='graph-combined', component_property='figure'),
        Output('graph-combined', 'style'),
        Output('graph_legend', 'children'),
        Output('loading-plot', 'children'),
        Input(component_id='plot-type', component_property='value'),
        Input(component_id='color-type', component_property='value'),
        Input('metadata-type', 'value'),
        Input('component_x', 'value'),
        Input('component_y', 'value'),
        Input('component_z', 'value'),
        Input('store-aligned', 'data'),
        Input('graph-combined', 'relayoutData'),
        State('data-selector', 'value'),
        State('session_id', 'data'),
        State('store-label-1', 'data'),
        State('store-label-2', 'data'),
        State('preprocess_1', 'value'),
        State('preprocess_2', 'value'),
        State('num_clusters', 'value'),
        State('num_enriched', 'value'),
        prevent_initial_call=True
    )
    def update_plot(plot_type, color_type, metadata_type, x, y, z, last_aligned, relayoutData,
                    dataset, session_id, label_1, label_2,
                    preprocess_1, preprocess_2, num_clusters, num_enriched):

        """Display visualization based on available data and selected visualization options
        """

        # Don't update the plot if alignment hasn't occurred
        if last_aligned == '0':
            raise PreventUpdate

        # Don't regenerate plot on relayoutData events unless 3-D separate plots (where using to synchronize cameras)
        if 'graph-combined.relayoutData' in ctx.triggered_prop_ids:
            if plot_type != 'separate3':
                raise PreventUpdate

        # Convert component numbers to integers and make 0-based instead of 1-based
        x = int(x) - 1
        y = int(y) - 1
        z = int(z) - 1

        # loading aligned data from cache
        df_1 = pd.read_json(cache.get(f'{session_id}-aligned_1'))
        df_1['cluster'] = df_1['cluster'].astype('string')
        Xg = np.array(df_1.drop(columns=['cluster', 'ttype']))

        df_2 = pd.read_json(cache.get(f'{session_id}-aligned_2'))
        df_2['cluster'] = df_2['cluster'].astype('string')
        Xe = np.array(df_2.drop(columns=['cluster', 'ttype']))

        # load metadata dataframe (use in conjunction with metadata-type)
        if metadata_type and dataset == 'upload':
            # uploaded metadata
            metadata_df = pd.read_json(cache.get(cache_key(session_id, UploadFileType.METADATA.name)))
            # append metadata to dataframes so that it's available for visualization
            df_1[metadata_type] = list(metadata_df[metadata_type])
            df_2[metadata_type] = list(metadata_df[metadata_type])


        style = {}  # default
        legend = ''
        if plot_type == 'alignment':
            fig = plot_alignment(df_1, df_2, label_1, label_2, dataset, x, y, z)
            style = {'height': '600px', 'width': '1000px'}
            legend = f"""{label_1} and {label_2} are projected to dimensions {x+1}, {y+1}, and {z+1} of the 
            latent space."""

        elif plot_type == 'alignment-error':
            fig = plot_alignment_error(df_1, df_2, dataset)
            style = {'height': '600px', 'width': '1000px'}
            legend = f"""Alignment error between the projections of {label_1} and {label_2} into the
            latent space.  Pairwise cell distance is the Euclidean distance between latent space projections
             for a single cell.  Fraction of Samples Closer Than True Match (FOSCTTM) is computed as follows.
             For each cell in {label_1}, we find its true match in {label_2}, then rank all other cells in the 
             latent space based on their distance from the first cell, finally computing the fraction of cells that are 
             closer than the true match.
             """
        elif plot_type == 'alignment-combo':
            fig = plot_alignment_and_error(df_1, df_2, label_1, label_2, dataset, x, y, z)
            style = {'height': '600px', 'width': '1000px'}
            legend = f"""Alignment error between the projections of {label_1} and {label_2} into the
            latent space.  Pairwise cell distance is the Euclidean distance between latent space projections
             for a single cell.  Fraction of Samples Closer Than True Match (FOSCTTM) is computed as follows.
             For each cell in {label_1}, we find its true match in {label_2}, then rank all other cells in the 
             latent space based on their distance from the first cell, finally computing the fraction of cells 
             that are closer than the true match.
             """

        elif plot_type == 'separate2':
            fig = scatter2d(df_1, df_2, x, y, color_type, metadata_type, label_1, label_2)
            style = {'height': '600px', 'width': '1000px'}
            legend = f"""{label_1} and {label_2} are separately projected to dimensions {x+1} and {y+1}
            of the latent space."""

        elif plot_type == 'separate3':
            fig = scatter3d(df_1, df_2, x, y, z, color_type, metadata_type, relayoutData, label_1, label_2)
            style = {'height': '600px', 'width': '1000px'}
            legend = f"""{label_1} and {label_2} are separately projected to dimensions {x+1}, {y+1}, and {z+1}
            of the latent space."""

        elif plot_type in ('bibiplot', 'heatmap', 'heatmap2'):
            # These plots need raw data, not just aligned data
            if dataset in ('motor', 'visual'):
                data_1 = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp.csv', index_col=0)
                data_2 = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature.csv', index_col=0)
                # if type(data_2.iloc[0, 0]) == str:
                #     # drop the first column if it contains strings (i.e., presumably cell names)
                #     data_2.drop(columns=data_2.columns[0], inplace=True)
            else:
                # get raw data from cache
                file_type_1 = UploadFileType.DATA_1.name
                file_type_2 = UploadFileType.DATA_2.name
                data_1 = pd.read_json(cache.get(cache_key(session_id, file_type_1)))
                data_2 = pd.read_json(cache.get(cache_key(session_id, file_type_2)))

                # Drop the first column, which is supposed to be cell names
                data_1.drop(columns=data_1.columns[0], inplace=True)
                data_2.drop(columns=data_2.columns[0], inplace=True)

            if plot_type == 'bibiplot':
                color_col = metadata_type if color_type == 'metadata' else color_type
                color_vec = df_1.get(color_col, None)
                fig = create_bibiplot1x2(data_1, data_2, Xg, Xe, x, y, dataset, color_vec, metadata_type,
                                         preprocess_1, preprocess_2,
                                         label_1, label_2)
                style = {'height': '600px', 'width': '1000px'}
                legend = f"""Biplots for {label_1} and {label_2} using dimensions {x+1} and {y+1} of the latent space.
                Features having a correlation with the latent space greater than 0.6 are shown plotted as radial lines 
                where the length is the value of correlation (max value 1). """

            elif plot_type == 'heatmap2':
                clusters = df_1['cluster']  # The clusters column is the same in df_1 and df_2
                fig, top_enriched = create_heatmap2(session_id, dataset, data_1, data_2, preprocess_1, preprocess_2,
                                                    clusters, num_clusters, label_1, label_2, num_enriched)
                legend = f"""Feature expression levels across all cells for the top {num_enriched} differentially-expressed features
                for each cross-modal cluster.  Clusters were identified by the Gaussian mixed model and normalized feature
                expression ranked using the Wilcox Rank Sum test."""

                # Store top-enriched to cache for subsequent download
                cache.set(f'{session_id}-enriched_1', top_enriched[1].to_json())
                cache.set(f'{session_id}-enriched_2', top_enriched[2].to_json())
        else:
            # No plot type specified - show a blank plot
            fig = go.Figure(data={}, layout=blank_layout)

        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

        return fig, style, legend, ''

    @app.callback(
        Output(component_id='loading-alignment', component_property='children'),
        Output('store-aligned', 'data'),
        Output('user-data-alert', 'children'),
        Output('user-data-alert', 'is_open'),
        Output('user-data-alert', 'color'),
        Output('metadata-type', 'options'),
        Input('session_id', 'data'),
        Input('btn-align', 'n_clicks'),
        Input('btn-cluster', 'n_clicks'),
        State('data-selector', 'value'),
        State('preprocess_1', 'value'),
        State('preprocess_2', 'value'),
        State('alignment-method', 'value'),
        State('ndims', 'value'),
        State('neighbors', 'value'),
        State('num_clusters', 'value'),
        State('clustering-method', 'value'),
        # State(component_id='eig-method', component_property='value'),
        # State(component_id='eig-count', component_property='value'),
        State({'type': 'dynamic-output', 'index': ALL}, 'children'),
        State('metadata-type', 'options'),
        prevent_initial_call=True
    )
    def align_and_cluster_datasets(session_id, align_clicks, cluster_clicks, dataset,
                                   preprocess_1, preprocess_2,
                                   alignment_method, ndims, neighbors,
                                   num_clusters, clustering_method,
                                   upload_filenames,
                                   metadata_options):

        error_msg = ''

        # If user-uploaded data, make sure that two data files have been uploaded (before trying to read from cache!)
        if dataset not in ('motor', 'visual'):
            if not (upload_filenames[0] and upload_filenames[1]):
                error_msg = 'Two data files must be uploaded before data can be aligned.'
                return '', '0', error_msg, bool(error_msg), 'danger', metadata_options

        # add logic for selectively processing only what has to be processed
        # (e.g., if clustering button is clicked, don't re-run alignment (which is expensive!)
        if ctx.triggered_id == 'btn-cluster':
            # load aligned data from cache
            try:
                df_1 = pd.read_json(cache.get(f'{session_id}-aligned_1'))
                df_2 = pd.read_json(cache.get(f'{session_id}-aligned_2'))
            except ValueError:
                error_msg = 'Data must be aligned before clusters can be identified.'
                return '', '0', error_msg, bool(error_msg), 'danger', metadata_options
            ttype = df_1['ttype']
            aligned_1 = np.array(df_1.drop(columns=['ttype', 'cluster']))
            aligned_2 = np.array(df_2.drop(columns=['ttype', 'cluster']))
        else:
            # perform dataset alignment and clustering
            if dataset in ('motor', 'visual'):
                # use mouse cortex data

                # Load mouse data and clean based on prior knowledge of structure.

                df_1 = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp.csv', index_col=0)
                X1 = np.array(df_1, dtype=float)

                df_2 = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature.csv', index_col=0)
                # drop the first column if it contains strings
                # if type(df_2.iloc[1, 0]) == str:
                #     X2 = np.array(df_2.iloc[:, 1:], dtype=float)
                # else:
                X2 = np.array(df_2, dtype=float)

                # Load metadata files containing ttype column
                ttype = pd.read_csv(f'data/mouse_{dataset}_cortex/metadata.csv')['ttype']

                # # temporary: write sample data files
                # df_1.to_csv(f'data/sample_geneExp.csv')
                # df_2.to_csv(f'data/sample_efeature.csv')
                # metadata = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_NMA_col.csv')
                # metadata.to_csv('data/sample_metadata.csv')

                metadata_options = {'ttype': 'Transcriptomic Type'}
            else:
                # load data from cache
                #try:
                # key_1 = cache_key(session_id, UploadFileType.DATA_1.name)
                # if cache.has(key_1):
                #     df_1 = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_1.name)))
                #     print(f'successfully got data from cache with key {key_1}')
                # else:
                #     print(f'Missing expected cache key {key_1}')
                #
                # key_2 = cache_key(session_id, UploadFileType.DATA_2.name)
                # if cache.has(key_2):
                #     df_2 = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_2.name)))
                #     print(f'successfully got data from cache with key {key_2}')
                # else:
                #     print(f'Missing expected cache key {key_2}')
                try:
                    df_1 = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_1.name)))
                    df_2 = pd.read_json(cache.get(cache_key(session_id, UploadFileType.DATA_2.name)))
                except ValueError:
                    error_msg = 'Two data files must be uploaded before data can be aligned.'
                    return '', '0', error_msg, bool(error_msg), 'danger', metadata_options

                if len(df_1) != len(df_2):
                    error_msg = 'Both data files must have the same number of rows.'
                    return '', '0', error_msg, bool(error_msg), 'danger', metadata_options

                # The first column is supposed to include cell identifiers, so drop it.
                X1 = np.array(df_1.iloc[:, 1:], dtype=float)
                X2 = np.array(df_2.iloc[:, 1:], dtype=float)

                if upload_filenames[2]:  # only try to read metadata from cache if a file has been uploaded
                    try:
                        metadata_df = pd.read_json(cache.get(cache_key(session_id, UploadFileType.METADATA.name)))
                        metadata_options = metadata_df.columns
                        ttype = None  #metadata_df['ttype']
                    except ValueError:
                        # Allow alignment to proceed even if no metadata file has been uploaded
                        ttype = None
                    except KeyError:
                        error_msg = 'The metadata file does not contain a column named "ttype".'
                        ttype = None
                else:
                    ttype = None

            # Apply selected preprocessing to raw datasets
            X1 = preprocess(X1, preprocess_1)
            X2 = preprocess(X2, preprocess_2)

            # Align datasets
            #proj, _ = nonlinear_manifold_alignment(X1, X2, int(ndims), int(neighbors)) #, eig_method=eig_method, eig_count=int(eig_count))
            try:
                proj = alignment(alignment_method, X1, X2, int(ndims), int(neighbors))
            except UnexpectedAlignmentMethodException:
                raise PreventUpdate

            aligned_1, aligned_2 = proj

        # Identify clusters (generalize later with clustering parameters, alternate methods)
        if clustering_method == 'gmm':
            clusters = cluster_gmm(aligned_1, aligned_2, num_clusters)
        elif clustering_method == 'kmeans':
            clusters = cluster_kmeans(aligned_1, aligned_2, num_clusters)
        elif clustering_method == 'hierarchical':
            clusters = cluster_hierarchical(aligned_1, aligned_2, num_clusters)
        else:
            # Should not happen
            clusters = None

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

        return '', str(time.time()), error_msg, bool(error_msg), 'warning', metadata_options


    @app.callback(
        Output('component_x', 'max'),
        Output('component_y', 'max'),
        Input('ndims', 'value')
    )
    def set_max_component(ndims):
        return ndims, ndims

    @app.callback(
        Output('plot-type', 'value'),
        Input('btn-align', 'n_clicks'),
        Input('btn-cluster', 'n_clicks')
    )
    def switch_visualization(align_clicks, cluster_clicks):
        """Change to the appropriate default visualization when clicking alignment or
        clustering button"""

        if ctx.triggered_id == 'btn-align':
            return 'alignment-combo'
        elif ctx.triggered_id == 'btn-cluster':
            return 'separate2'
        else:
            raise PreventUpdate


    @app.callback(
        Output('download-aligned', 'data'),
        Input('btn-align-download', 'n_clicks'),
        State('session_id', 'data'),
        State('store-label-1', 'data'),
        State('store-label-2', 'data'),
        prevent_initial_call=True,
    )
    def download_aligned(n_clicks, session_id, label_1, label_2):
        """ Download aligned data.  Since there are two dataframes, the best way to handle this is probably to download
        a .zip file containing two .csv files.
        """

        # read data frames from cache
        try:
            df_1 = pd.read_json(cache.get(f'{session_id}-aligned_1'))
            df_2 = pd.read_json(cache.get(f'{session_id}-aligned_2'))
        except ValueError:
            raise PreventUpdate

        # These dataframes include the aligned data as well as the cluster and ttype
        # df_1.drop(columns=['cluster', 'ttype'])
        # df_2.drop(columns=['cluster', 'ttype'])

        # make labels safe to use in filenames (and distinct)
        labels = safe_filenames(label_1, label_2)

        bytes_io = io.BytesIO()
        with zipfile.ZipFile(bytes_io, mode='w') as zf:
            zf.writestr(f'aligned_{labels[0]}.csv', df_1.to_csv())
            zf.writestr(f'aligned_{labels[1]}.csv', df_2.to_csv())

        return dcc.send_bytes(bytes_io.getvalue(), 'aligned.zip', type='application/zip')

    @app.callback(
        Output('download-cluster', 'data'),
        Input('btn-cluster-download', 'n_clicks'),
        State('session_id', 'data'),
        State('store-label-1', 'data'),
        State('store-label-2', 'data'),
        prevent_initial_call=True,
    )
    def download_clusters(n_clicks, session_id, label_1, label_2):
        """ Download cluster data.  Since there are two dataframes, the best way to handle this is probably to download
        a .zip file containing two .csv files.
        """

        # read data frames from cache
        try:
            df_1 = pd.read_json(cache.get(f'{session_id}-aligned_1'))
            df_2 = pd.read_json(cache.get(f'{session_id}-aligned_2'))
        except ValueError:
            raise PreventUpdate

        # These dataframes include the aligned data as well as the cluster and ttype
        # df_1.drop(columns=['cluster', 'ttype'])
        # df_2.drop(columns=['cluster', 'ttype'])

        # make labels safe to use in filenames (and distinct)
        labels = safe_filenames(label_1, label_2)

        bytes_io = io.BytesIO()
        with zipfile.ZipFile(bytes_io, mode='w') as zf:
            zf.writestr(f'clusters_{labels[0]}.csv', df_1.to_csv())
            zf.writestr(f'clusters_{labels[1]}.csv', df_2.to_csv())

        return dcc.send_bytes(bytes_io.getvalue(), 'clusters.zip', type='application/zip')

    @app.callback(
        Output('download-enriched', 'data'),
        Input('btn-enriched-download', 'n_clicks'),
        State('session_id', 'data'),
        State('store-label-1', 'data'),
        State('store-label-2', 'data'),
        prevent_initial_call=True,
    )
    def download_enriched(n_clicks, session_id, label_1, label_2):
        """ Download enriched data.  Since there are two dataframes, the best way to handle this is probably to download
        a .zip file containing two .csv files.
        """

        # read data frames from cache
        try:
            df_1 = pd.read_json(cache.get(f'{session_id}-enriched_1'))
            df_2 = pd.read_json(cache.get(f'{session_id}-enriched_2'))
        except ValueError:
            raise PreventUpdate

        # make labels safe to use in filenames (and distinct)
        labels = safe_filenames(label_1, label_2)

        bytes_io = io.BytesIO()
        with zipfile.ZipFile(bytes_io, mode='w') as zf:
            zf.writestr(f'enriched_{labels[0]}.csv', df_1.to_csv())
            zf.writestr(f'enriched_{labels[1]}.csv', df_2.to_csv())

        return dcc.send_bytes(bytes_io.getvalue(), 'enriched.zip', type='application/zip')

    @app.callback(
        Output('download-sample-data', 'data'),
        Input('sample-data-download-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_sample_data(n_clicks):
        return dcc.send_file('./assets/mouse_motor_cortex.zip')


    @app.callback(
        Output('about-page', 'className'),
        Output('upload-page', 'className'),
        Output('graph-page', 'className'),
        Input('left-panel-tabs', 'value'),
        prevent_initial_call=True,
    )
    def handle_tab_change(selected_tab):
        """Control visibility of right-pane content based on selected tab"""
        visibility = ['hidden'] * 3
        if selected_tab == 'tab-1':
            visibility[0] = ''
        elif selected_tab == 'tab-2':
            visibility[1] = ''
        else:
            visibility[2] = ''
        return visibility

    @app.callback(
        Output('left-panel-tabs', 'value'),
        Input({'type': 'next-button', 'index': ALL}, 'n_clicks'),
        State('left-panel-tabs', 'value')
    )
    def handle_next_button(n_clicks, selected_tab):
        if selected_tab == 'tab-2':
            return 'tab-3'
        elif selected_tab == 'tab-3':
            return 'tab-4'
        elif selected_tab == 'tab-4':
            return 'tab-5'
        else:
            raise PreventUpdate


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



