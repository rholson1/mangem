from dash import html, dcc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import uuid

from application.settings import appname
from application.constants import UploadFileType, blank_layout, color_types


def get_layout():
    session_id = str(uuid.uuid4())

    layout = html.Div(children=[
        dcc.Store(id='session_id', data=session_id),
        dcc.Store(id='store-data_selected', data='0'),
        dcc.Store(id='store-aligned', data='0'),
        dcc.Store(id='store-label-1', data='Modality 1'),
        dcc.Store(id='store-label-2', data='Modality 2'),

        html.Div(id='id_header',
                 children=[
                     html.H1(children=appname),
                 ]),

    #     html.Div(children='''
    #     Demonstration of plotting aligned data in latent space (first three dimensions).
    # '''),

        html.Div(
            id='main-block',
            children=[
                html.Div(
                    id='left-panel',
                    children=[
                        dcc.Tabs(
                            id='left-panel-tabs',
                            className='custom-tabs-container',
                            children=[
                                dcc.Tab(
                                    label='About',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    children=[
                                        html.H3(f'About {appname}'),
                                        html.Img(src='assets/cover_figure.png',
                                                 alt='Manifold alignment and clustering flow chart',
                                                 className='tab-image'),
                                        html.P(children=[
                                            'This application demonstrates use of nonlinear manifold learning to '
                                            'align multimodal single-cell data sets in a low-dimension latent space, '
                                            'enabling identification of cross-modal cell clusters.  '
                                            # 'This application aligns multimodal single-cell data sets to support '
                                            # 'developing a deeper understanding of neural function and features. '
                                            'The application implements methods described in ',
                                            html.A(href='https://doi.org/10.1038/s42003-021-02807-6',
                                                   children=('Manifold learning analysis suggests strategies for aligning '
                                                      'single-cell multi-modalities and revealing functional genomics '
                                                      'for neuronal electrophysiology'),
                                                   target='_blank'),
                                            '  and includes datasets from that paper (mouse visual cortex and motor cortex) '
                                            'while also supporting use of uploaded data.'
                                        ]),
                                        html.H3('Usage'),
                                        html.Ol(children=[
                                            html.Li([
                                                html.H4('Select Data'),
                                                html.P('On the Data tab, select a predefined data set (mouse motor or '
                                                       'visual cortex neurons) or upload your own data files. '
                                                       'User uploaded data must be in .csv files where the first row '
                                                       'contains column names and the first column contains cell '
                                                       'identifiers.  A data set includes two files, one for each modality '
                                                       '(e.g., gene expression, cell morphology, or electrophysiological '
                                                       'features), along with an optional metadata file that identifies '
                                                       'cell transcriptomic type (with column header "ttype"). '
                                                       'Files must have consistent cell (row) order.'
                                                       ),
                                                html.P('Currently, only the first 1000 rows of uploaded data files are used.'),
                                                html.H5('Preprocessing'),
                                                html.P('Optionally, select a preprocessing operation to be performed '
                                                       'on each modality prior to alignment.  The currently supported '
                                                       'operations are log transformation and standardization.')]),
                                            html.Li([
                                                html.H4('Align Data'),
                                                html.P('The alignment tab provides controls to select the dimension '
                                                       'of the latent space to which the cellular data should be projected '
                                                       'as well as the number of nearest neighbors to be used in constructing '
                                                       'similarity matrices.'),
                                                html.P('Click the "Align Datasets" button to perform the alignment. '
                                                       'If the alignment parameters are changed, the "Align Datasets" '
                                                       'button will need to be used again to recompute the alignment.'),
                                                html.P('Once the alignment has been performed, the aligned data can be '
                                                       'downloaded by clicking on the "Download Aligned Data" button.')]),
                                            html.Li([
                                                html.H4('Clustering'),
                                                html.P('The clustering tab includes a control to select the number of '
                                                       'cross-modal clusters which should be identified.  Click the '
                                                       '"Download Clusters" button to download files that label '
                                                       'cells with the identified cross-modal clusters.')
                                            ]),
                                            html.Li([
                                                html.H4('Cluster Analysis'),
                                                html.P('The cluster analysis tab allows selection of visualizations '
                                                       'and control of relevant parameters. '
                                                       'Supported plot types include:'),
                                                html.Ul([
                                                    html.Li([
                                                        html.H5('Dataset alignment'),
                                                        html.Span('Display a 3-D projection of the aligned datasets '
                                                                  ' overlaid in the latent space.')
                                                    ]),
                                                    html.Li([
                                                        html.H5('Alignment error metrics'),
                                                        html.Span('Display box plots characterizing alignment error '
                                                                  'using pairwise cell distance and Fraction of Samples '
                                                                  'Closer Than True Match (FOSCTTM).')
                                                    ]),
                                                    html.Li([
                                                        html.H5('Separate 2-D plots'),
                                                        html.Span('Display a 2-D projection of the aligned datasets '
                                                                  'in two side-by-side plots.')
                                                    ]),
                                                    html.Li([
                                                        html.H5('Separate 3-D plots'),
                                                        html.Span('Display a 3-D projection of the aligned datasets '
                                                                  'in two side-by-side plots.')
                                                    ]),
                                                    html.Li([
                                                        html.H5('Bibiplot'),
                                                        html.Span('Display biplots in two dimensions of the latent space. '
                                                                  'Aligned datasets are projected together with lines '
                                                                  'describing the correlation of features with the '
                                                                  'latent space.')
                                                    ]),
                                                    html.Li([
                                                        html.H5('Feature enrichment by cluster'),
                                                        html.Span('Computes and plots the top differentially-expressed '
                                                                  'features by cross-modal cluster.')
                                                    ]),

                                                ]),
                                                html.P('Controls are provided to control cell coloring (by cross-modal '
                                                       'cluster or transcriptomic type, if available) and to select the '
                                                       'latent space components which should be used in creating the '
                                                       'above visualizations.')

                                            ]),
                                            html.H4('Top Feature Enrichment By Cluster '),
                                            html.P('The feature enrichment by cluster plot computes the most enriched '
                                                   'features for each cross-modal cluster.  The number of features to '
                                                   'be identified can be specified, and after the plot has been '
                                                   'generated, the list of most-enriched features for each cluster can '
                                                   'be downloaded by clicking the "Download Most Enriched Features" '
                                                   'button.')
                                        ])
                                    ]
                                ),

                                dcc.Tab(
                                    label='Data',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    id='data-block',
                                    children=[
                                        html.H3('Data', className='block-title'),
                                        html.H4([
                                            'Select Data',
                                            html.I(id='data-select-tooltip', className='fa-regular fa-circle-question')
                                        ]),
                                        dbc.Tooltip(
                                            'Select an existing dataset or upload your own .csv files.',
                                            target='data-select-tooltip'
                                        ),
                                        dcc.Dropdown(
                                            id='data-selector',
                                            options={
                                                'motor': 'Mouse Motor Cortex',
                                                'visual': 'Mouse Visual Cortex',
                                                'upload': 'Upload your data!'
                                            },
                                            value=''
                                        ),

                                        html.Div(
                                            id='upload-container',
                                            className='hidden',
                                            children=[

                                                html.Div(
                                                    className='upload-block',
                                                    children=[
                                                        html.H4([
                                                            'Modality 1',
                                                            html.I(id='modality-1-tooltip',
                                                                   className='fa-regular fa-circle-question')
                                                        ]),
                                                        dbc.Tooltip(
                                                            'All data should be uploaded in .csv files.  '
                                                            'Column headers must be present.  The first column should '
                                                            'contain cell identifiers.  All data files should have the '
                                                            'same number of rows and should have a consistent row order '
                                                            'corresponding to the cells.',
                                                            target='modality-1-tooltip'
                                                        ),
                                                        dcc.Input(
                                                            id='upload_1_label',
                                                            type='text',
                                                            value='',
                                                            placeholder='Modality 1 Label'
                                                        ),
                                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': UploadFileType.DATA_1},
                                                                   children=html.Button('Upload Data'),
                                                                   className='upload'),
                                                        html.Div(id={'type': 'dynamic-upload-info',
                                                                     'index': UploadFileType.DATA_1}),
                                                        html.Div(id={'type': 'dynamic-output', 'index': UploadFileType.DATA_1}),
                                                        # dcc.Upload(id={'type': 'dynamic-upload', 'index': 2},
                                                        #            children=html.Button('Upload Metadata'),
                                                        #            className='upload'),
                                                        # html.Div(id={'type': 'dynamic-output', 'index': 2})
                                                    ]
                                                ),
                                                html.Div(
                                                    className='upload-block',
                                                    children=[
                                                        html.H4([
                                                            'Modality 2',
                                                            html.I(id='modality-2-tooltip',
                                                                   className='fa-regular fa-circle-question')
                                                        ]),
                                                        dbc.Tooltip(
                                                            'All data should be uploaded in .csv files.  '
                                                            'Column headers must be present.  The first column should '
                                                            'contain cell identifiers.  All data files should have the '
                                                            'same number of rows and should have a consistent row order '
                                                            'corresponding to the cells.',
                                                            target='modality-2-tooltip'
                                                        ),
                                                        dcc.Input(
                                                            id='upload_2_label',
                                                            type='text',
                                                            value='',
                                                            placeholder='Modality 2 Label'
                                                        ),
                                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': UploadFileType.DATA_2},
                                                                   children=html.Button('Upload Data'),
                                                                   className='upload'),
                                                        html.Div(id={'type': 'dynamic-upload-info',
                                                                     'index': UploadFileType.DATA_2}),
                                                        html.Div(id={'type': 'dynamic-output', 'index': UploadFileType.DATA_2}),
                                                        # dcc.Upload(id={'type': 'dynamic-upload', 'index': 4},
                                                        #            children=html.Button('Upload Metadata'),
                                                        #            className='upload'),
                                                        # html.Div(id={'type': 'dynamic-output', 'index': 4})
                                                    ]
                                                ),
                                                html.Div(
                                                    className='upload-block',
                                                    children=[
                                                        html.H4([
                                                            'Metadata',
                                                            html.I(id='metadata-tooltip',
                                                                   className='fa-regular fa-circle-question')
                                                        ]),
                                                        dbc.Tooltip(
                                                            'Additional information can be supplied in a metadata file. '
                                                            'It should be a .csv file with column headers and one row '
                                                            'per cell.  Only "ttype" is currently supported.',
                                                            target='metadata-tooltip'
                                                        ),
                                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': UploadFileType.METADATA},
                                                                   children=html.Button('Upload Metadata'),
                                                                   className='upload'),
                                                        html.Div(id={'type': 'dynamic-upload-info',
                                                                     'index': UploadFileType.METADATA}),
                                                        html.Div(id={'type': 'dynamic-output', 'index': UploadFileType.METADATA})
                                                    ]
                                                ),
                                            ]
                                        ),
                                        html.H4([
                                            'Preprocessing',
                                            html.I(id='preprocessing-tooltip', className='fa-regular fa-circle-question')
                                        ]),
                                        dbc.Tooltip(
                                            'Each dataset may optionally be transformed prior to alignment.',
                                            target='preprocessing-tooltip'
                                        ),
                                        html.Label([
                                            html.Span('', id='preprocess-label-1'),
                                            ': ',
                                            dcc.Dropdown(
                                                id='preprocess_1',
                                                options={
                                                    'scale': 'Standardize',
                                                    'log': 'Log transform'
                                                },
                                                value='log'
                                            )
                                        ]),
                                        html.Label([
                                            html.Span('', id='preprocess-label-2'),
                                            ': ',
                                            dcc.Dropdown(
                                                id='preprocess_2',
                                                options={
                                                    'scale': 'Standardize',
                                                    'log': 'Log transform'
                                                },
                                                value='scale'
                                            )
                                        ]),

                                    ]
                                ),
                                dcc.Tab(
                                    label='Alignment',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    id='alignment-block',
                                    children=[
                                        html.Div([
                                            html.H3('Dataset Alignment', className='block-title'),
                                            html.Label([
                                                'Latent space dimension: ',
                                                # dcc.Input(id='ndims', value='5', style={'width': '20px'},
                                                #           type='number', min=3, max=20, step=1),
                                                dcc.Slider(id='ndims', value=5, min=3, max=10, step=1)
                                            ]),
                                            html.Br(),
                                            html.Label([
                                                'Number of nearest neighbors: ',
                                                # dcc.Input(id='neighbors', value='2', style={'width': '20px'},
                                                #           type='number', min=1, max=10, step=1),
                                                dcc.Slider(id='neighbors', value=2, min=1, max=10, step=1)
                                            ]),
                                            html.Button('Align Datasets', id='btn-align',
                                                        className='operations-button'),
                                            html.Button('Download Aligned Data', id='btn-align-download',
                                                        className='operations-button'),
                                            dcc.Download(id='download-aligned')
                                        ])
                                    ]
                                ),

                                dcc.Tab(
                                    label='Clustering',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    id='clustering-block',
                                    children=[
                                        html.Div([
                                            html.H3('Cross-modal clustering', className='block-title'),
                                            # html.P(id='alignment-state', className='status_message'),
                                            # html.Button('Align Datasets', id='btn-align', className='operations-button'),
                                            # html.Details([
                                            #     html.Summary('Alignment parameters'),
                                            #     html.Div([
                                            #         # html.Label(
                                            #         #     children=[
                                            #         #         'eigenvalue method: ',
                                            #         #         dcc.RadioItems(
                                            #         #             id='eig-method',
                                            #         #             options={
                                            #         #                 'eig': 'eig',
                                            #         #                 'eigs': 'eigs',
                                            #         #                 'eigsh': 'eigsh'
                                            #         #             },
                                            #         #             value='eigs'
                                            #         #         ),
                                            #         #     ]
                                            #         # ),
                                            #         # html.Br(),
                                            #
                                            #     ]),
                                            # ]),

                                            html.Label(
                                                children=[
                                                    'Clustering Algorithm: ',
                                                    dcc.RadioItems(
                                                        id='clustering-method',
                                                        options={
                                                            'gmm': 'Gaussian Mixture Model',
                                                        },
                                                        value='gmm'
                                                    ),
                                                ]
                                            ),
                                            html.Br(),
                                            html.Label([
                                                'Number of clusters: ',
                                                # dcc.Input(id='num_clusters', value='5', style={'width': '20px'},
                                                #           type='number', min=1, max=10, step=1),
                                                dcc.Slider(id='num_clusters', value=5, min=1, max=10, step=1)
                                            ]),
                                            html.Button('Identify Clusters', id='btn-cluster', className='operations-button'),
                                            html.Button('Download Clusters', id='btn-cluster-download',
                                                        className='operations-button'),
                                            dcc.Download(id='download-cluster'),
                                        ])
                                    ]
                                ),

                                dcc.Tab(
                                    label='Cluster Analysis',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    id='controls-block',
                                    children=[
                                        html.H3('Visualization', className='block-title'),
                                        html.Div([
                                            html.H4('Plot type'),
                                            dcc.Dropdown(
                                                id='plot-type',
                                                options={
                                                    'alignment': 'Dataset alignment',
                                                    'alignment-error': 'Alignment error metrics',
                                                    'separate2': 'Separate 2-D plots',
                                                    'separate3': 'Separate 3-D plots',
                                                    'bibiplot': 'Bibiplot',
                                                    #'heatmap': 'Gene enrichment by cluster [Mouse cortex]',
                                                    'heatmap2': 'Feature enrichment by cluster'
                                                },
                                                value='alignment'
                                            ),

                                            html.H4('Color by...'),
                                            dcc.RadioItems(
                                                id='color-type',
                                                options=color_types,
                                                value='cluster'
                                            ),
                                            html.H4('Component Selection',
                                                    title='Select components of the aligned datasets to be plotted .'),
                                            # html.Dialog()
                                            html.Label([' X: ',
                                                        dcc.Input(id='component_x',
                                                                  type='number',
                                                                  min=1,
                                                                  step=1,
                                                                  value=1,
                                                                  className='component_input')]),
                                            html.Label([' Y: ',
                                                        dcc.Input(id='component_y',
                                                                  type='number',
                                                                  min=1,
                                                                  step=1,
                                                                  value=2,
                                                                  className='component_input')]),
                                            html.Label([' Z: ',
                                                        dcc.Input(id='component_z',
                                                                  type='number',
                                                                  min=1,
                                                                  step=1,
                                                                  value=3,
                                                                  className='component_input')]),
                                            html.H3('Top Feature Enrichment by Cluster'),
                                            html.H4('Most-enriched features per cluster to identify'),
                                            dcc.Input(id='num_enriched',
                                                      type='number',
                                                      min=1,
                                                      step=1,
                                                      value=10,
                                                      className='component_input'),
                                            html.Button('Download Most Enriched Features', id='btn-enriched-download',
                                                        className='operations-button'),
                                            dcc.Download(id='download-enriched'),
                                        ])
                                    ]
                                )
                            ]
                        ),
                    ]

                ),



                html.Div(
                    id='right-panel',
                    children=[

                        html.Div(
                            id='graph_block',
                            children=[
                                dcc.Graph(
                                    id='graph-combined',
                                    mathjax=True,
                                    figure=go.Figure(data={}, layout=blank_layout),
                                    style={'height': '600px', 'width': '600px'}
                                ),
                            ]
                        ),
                        html.P(id='graph_legend', children='')

                    ]
                ),
                html.Div(id='progress_container', children=[
                    dcc.Loading(id='loading-1',
                                children=[html.Div(id='loading-alignment'),
                                          html.Div(id='loading-plot'),
                                          html.Div(id={'type': 'loading-upload', 'index': UploadFileType.DATA_1}),
                                          html.Div(id={'type': 'loading-upload', 'index': UploadFileType.DATA_2}),
                                          html.Div(id={'type': 'loading-upload', 'index': UploadFileType.METADATA}),
                                          ]),
                    # dcc.Loading(id='loading-2',
                    #             children=html.Div(id='loading-plot', className='loading_trigger')),
                    # id={'type': 'dynamic-output', 'index': UploadFileType.DATA_2}
                ])

            ]
        )

    ])

    return layout
