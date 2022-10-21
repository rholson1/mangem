from dash import html, dcc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import uuid

from application.settings import appname, color_types

blank_layout = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'xaxis': {'visible': False, 'showline': False, 'showgrid': False},
    'yaxis': {'visible': False, 'showline': False, 'showgrid': False},
}


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
                # html.Div(
                #     id='left-panel',
                #     children=[
                #
                #         html.Div(
                #             id='data-block',
                #             children=[
                #                 html.H3('Data', className='block-title'),
                #                 html.P('Upload datasets to be aligned, optionally with metadata.'),
                #
                #                 html.H4('Select Data'),
                #                 dcc.Dropdown(
                #                     id='data-selector',
                #                     options={
                #                         'motor': 'scMNC Mouse Motor Cortex',
                #                         'visual': 'scMNC Mouse Visual Cortex',
                #                         'upload': 'Upload your data!'
                #                     },
                #                     value=''
                #                 ),
                #
                #                 html.Div(
                #                     id='upload-container',
                #                     className='hidden',
                #                     children=[
                #                         html.Div(
                #                             className='upload-block',
                #                             children=[
                #                                 dcc.Input(
                #                                     id='upload_1_label',
                #                                     type='text',
                #                                     placeholder='File 1 Label'
                #                                 ),
                #                                 dcc.Upload(id={'type': 'dynamic-upload', 'index': 1},
                #                                            children=html.Button('Upload File 1'), className='upload'),
                #                                 html.Div(id={'type': 'dynamic-output', 'index': 1}),
                #                                 dcc.Upload(id={'type': 'dynamic-upload', 'index': 2},
                #                                            children=html.Button('Metadata'), className='upload'),
                #                                 html.Div(id={'type': 'dynamic-output', 'index': 2})
                #                             ]
                #                         ),
                #                         html.Div(
                #                             className='upload-block',
                #                             children=[
                #                                 dcc.Input(
                #                                     id='upload_2_label',
                #                                     type='text',
                #                                     placeholder='File 2 Label'
                #                                 ),
                #                                 dcc.Upload(id={'type': 'dynamic-upload', 'index': 3},
                #                                            children=html.Button('Upload File 2'), className='upload'),
                #                                 html.Div(id={'type': 'dynamic-output', 'index': 3}),
                #                                 dcc.Upload(id={'type': 'dynamic-upload', 'index': 4},
                #                                            children=html.Button('Metadata'), className='upload'),
                #                                 html.Div(id={'type': 'dynamic-output', 'index': 4})
                #                             ]
                #                         ),
                #                     ]
                #                 ),
                #                 html.H4('Preprocessing'),
                #                 html.Label([
                #                     'Electrophys: ',
                #                     dcc.Dropdown(
                #                         id='preprocess_1',
                #                         options={
                #                             'scale': 'Standardize',
                #                             'log': 'Log transform'
                #                         },
                #                         value='scale'
                #                     )
                #                 ]),
                #                 html.Label([
                #                     'Gene Expression: ',
                #                     dcc.Dropdown(
                #                         id='preprocess_2',
                #                         options={
                #                             'scale': 'Standardize',
                #                             'log': 'Log transform'
                #                         },
                #                         value='log'
                #                     )
                #                 ]),
                #
                #             ]
                #         ),
                #
                #         html.Div(
                #             id='operations-block',
                #             children=[
                #                 html.H3('Dataset Alignment', className='block-title'),
                #                 html.Div([
                #                     html.P(id='alignment-state', className='status_message'),
                #                     html.Button('Align Datasets', id='btn-align'),
                #                     html.Details([
                #                         html.Summary('Alignment parameters'),
                #                         html.Div([
                #                             html.Label(
                #                                 children=[
                #                                     'eigenvalue method: ',
                #                                     dcc.RadioItems(
                #                                         id='eig-method',
                #                                         options={
                #                                             'eig': 'eig',
                #                                             'eigs': 'eigs',
                #                                             'eigsh': 'eigsh'
                #                                         },
                #                                         value='eigs'
                #                                     ),
                #                                 ]
                #                             ),
                #                             html.Br(),
                #                             html.Label([
                #                                 '# eigenvalues: ',
                #                                 dcc.Input(id='eig-count', value='5', style={'width': '20px'}),
                #                             ]),
                #                         ]),
                #                     ]),
                #                     html.Button('Identify Clusters', id='btn-cluster'),
                #                     html.Details([
                #                         html.Summary('Clustering parameters'),
                #                         html.Div([
                #                             html.Label(
                #                                 children=[
                #                                     'clustering method: ',
                #                                     dcc.RadioItems(
                #                                         id='clustering-method',
                #                                         options={
                #                                             'gmm': 'GMM',
                #                                         },
                #                                         value='gmm'
                #                                     ),
                #                                 ]
                #                             ),
                #                         ]),
                #                     ]),
                #                     html.Div(id='progress_container', children=[
                #                         dcc.Loading(id='loading-1', children=html.Div(id='loading-output-1') ),
                #                         html.Br(),
                #                         dcc.Loading(id='loading-2', children=html.Div(id='loading-output-2')),
                #                     ])
                #                 ])
                #             ]
                #         ),
                #
                #         html.Div(
                #             id='controls-block',
                #             children=[
                #                 html.H3('Visualization', className='block-title'),
                #                 html.Div([
                #                     html.H4('Plot type'),
                #                     dcc.Dropdown(
                #                         id='plot-type',
                #                         options={
                #                             'alignment': 'Dataset alignment',
                #                             'alignment-error': 'Alignment error',
                #                             'separate2': 'Separate 2-D plots',
                #                             'separate3': 'Separate 3-D plots',
                #                             'bibiplot': 'Bibiplot',
                #                             'heatmap': 'Gene enrichment by cluster'
                #                         },
                #                         value='alignment'
                #                     ),
                #
                #                     html.H4('Color by...'),
                #                     dcc.RadioItems(
                #                         id='color-type',
                #                         options=color_types,
                #                         value='cluster'
                #                     ),
                #                     html.H4('Component Selection', title='Select components of the aligned datasets to be plotted .'),
                #                     #html.Dialog()
                #                     html.Label([' X: ',
                #                                 dcc.Input(id='component_x',
                #                                           type='number',
                #                                           min='1',
                #                                           step='1',
                #                                           value='1',
                #                                           className='component_input')]),
                #                     html.Label([' Y: ',
                #                                 dcc.Input(id='component_y',
                #                                           type='number',
                #                                           min='1',
                #                                           step='1',
                #                                           value='2',
                #                                           className='component_input')]),
                #                     html.Label([' Z: ',
                #                                 dcc.Input(id='component_z',
                #                                           type='number',
                #                                           min='1',
                #                                           step='1',
                #                                           value='3',
                #                                           className='component_input')])
                #
                #                 ])
                #             ]
                #         )
                #     ]
                # ),
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
                                        html.P(children=[
                                            'This application aligns multimodal single-cell data sets to support '
                                            'developing a deeper understanding of neural function and features. '
                                            'Datasets used in ',
                                            html.A(href='https://doi.org/10.1038/s42003-021-02807-6',
                                                   children=('Manifold learning analysis suggests strategies for aligning '
                                                      'single-cell multi-modalities and revealing functional genomics '
                                                      'for neuronal electrophysiology'),
                                                   target='_blank'),
                                            ' are provided (mouse visual cortext or mouse motor cortex), '
                                            'or user data may be uploaded.'
                                        ]),



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
                                                            placeholder='Modality 1 Label'
                                                        ),
                                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': 1},
                                                                   children=html.Button('Upload Data'),
                                                                   className='upload'),
                                                        html.Div(id={'type': 'dynamic-output', 'index': 1}),
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
                                                            placeholder='Modality 2 Label'
                                                        ),
                                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': 3},
                                                                   children=html.Button('Upload Data'),
                                                                   className='upload'),
                                                        html.Div(id={'type': 'dynamic-output', 'index': 3}),
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
                                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': 4},
                                                                   children=html.Button('Upload Metadata'),
                                                                   className='upload'),
                                                        html.Div(id={'type': 'dynamic-output', 'index': 4})
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
                                                value='scale'
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
                                                value='log'
                                            )
                                        ]),

                                    ]
                                ),

                                dcc.Tab(
                                    label='Alignment',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    id='operations-block',
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
                                        ]),



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
                                            # html.Details([
                                            #     html.Summary('Clustering parameters'),
                                            #     html.Div([
                                            #         html.Label(
                                            #             children=[
                                            #                 'clustering method: ',
                                            #                 dcc.RadioItems(
                                            #                     id='clustering-method',
                                            #                     options={
                                            #                         'gmm': 'GMM',
                                            #                     },
                                            #                     value='gmm'
                                            #                 ),
                                            #             ]
                                            #         ),
                                            #     ]),
                                            # ]),
                                            html.Div(id='progress_container', children=[
                                                dcc.Loading(id='loading-1', children=html.Div(id='loading-output-1')),
                                                html.Br(),
                                                dcc.Loading(id='loading-2', children=html.Div(id='loading-output-2')),
                                            ])
                                        ])
                                    ]
                                ),

                                dcc.Tab(
                                    label='Plots',
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
                                                    'heatmap': 'Gene enrichment by cluster'
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
                                                                  min='1',
                                                                  step='1',
                                                                  value='1',
                                                                  className='component_input')]),
                                            html.Label([' Y: ',
                                                        dcc.Input(id='component_y',
                                                                  type='number',
                                                                  min='1',
                                                                  step='1',
                                                                  value='2',
                                                                  className='component_input')]),
                                            html.Label([' Z: ',
                                                        dcc.Input(id='component_z',
                                                                  type='number',
                                                                  min='1',
                                                                  step='1',
                                                                  value='3',
                                                                  className='component_input')])

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
                )
            ]
        )

        # html.Div(children=[
        #     html.Div(id='controls_container', children=[
        #         html.P("[Inactive controls]"),
        #         dcc.Upload(
        #             id='upload_1',
        #             children=html.Div([
        #                 'Drag and Drop or ',
        #                 html.A('Select File 1')
        #             ]),
        #             className='upload-button',
        #             # Allow multiple files to be uploaded
        #             multiple=False
        #         ),
        #         dcc.Input(
        #             id='upload_1_label',
        #             type='text',
        #             placeholder='File 1 Label'
        #         ),
        #
        #         dcc.Upload(
        #             id='upload_2',
        #             children=html.Div([
        #                 'Drag and Drop or ',
        #                 html.A('Select File 2')
        #             ]),
        #             className='upload-button',
        #             # Allow multiple files to be uploaded
        #             multiple=False
        #         ),
        #         dcc.Input(
        #             id='upload_2_label',
        #             type='text',
        #             placeholder='File 2 Label'
        #         ),
        #
        #     ], style={'flex': 1}),
        #     html.Div(id='graph_container', children=[
        #         dcc.Graph(
        #             id='graph_combined',
        #             figure=fig_combined,
        #             style={'height': '600px', 'width': '100%'}
        #         ),
        #     ], style={'flex': 1})
        # ], style={
        #     'display': 'flex',
        #     'flexDirection': 'row',
        #     'border': '1.5px solid gray',
        #     'border-radius': '5px',
        #     'padding': '10px'
        # })

        # dcc.Graph(
        #     id='graph_combined',
        #     figure=fig_combined,
        # ),

        # dcc.Graph(
        #     id='graph_electrophys',
        #     figure=fig_e
        # ),

        # dcc.Graph(
        #     id='graph_geneexpression',
        #     figure=fig_g
        # )
    ])

    return layout
