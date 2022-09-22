from dash import html, dcc
import plotly.graph_objects as go
import uuid

color_types = {'ttype': 't-type', 'gmm_cluster': 'Cluster'}


def get_layout():
    session_id = str(uuid.uuid4())

    layout = html.Div(children=[
        dcc.Store(data=session_id, id='session_id'),

        html.H1(children='BRAIN - Multimodal Alignment'),

        html.Div(children='''
        Demonstration of plotting aligned data in latent space (first three dimensions).
    '''),

        html.Div(
            id='main-block',
            children=[
                html.Div(
                    id='left-panel',
                    children=[

                        html.Div(
                            id='data-block',
                            children=[
                                html.H3('Data', className='block-title'),
                                html.P('Upload datasets to be aligned, optionally with metadata.'),
                                html.Div(
                                    className='upload-block',
                                    children=[
                                        dcc.Input(
                                            id='upload_1_label',
                                            type='text',
                                            placeholder='File 1 Label'
                                        ),
                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': 1},
                                                   children=html.Button('Upload File 1'), className='upload'),
                                        html.Div(id={'type': 'dynamic-output', 'index': 1}),
                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': 2},
                                                   children=html.Button('Metadata'), className='upload'),
                                        html.Div(id={'type': 'dynamic-output', 'index': 2})
                                    ]
                                ),
                                html.Div(
                                    className='upload-block',
                                    children=[
                                        dcc.Input(
                                            id='upload_2_label',
                                            type='text',
                                            placeholder='File 2 Label'
                                        ),
                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': 3},
                                                   children=html.Button('Upload File 2'), className='upload'),
                                        html.Div(id={'type': 'dynamic-output', 'index': 3}),
                                        dcc.Upload(id={'type': 'dynamic-upload', 'index': 4},
                                                   children=html.Button('Metadata'), className='upload'),
                                        html.Div(id={'type': 'dynamic-output', 'index': 4})
                                    ]
                                ),

                                html.H4('Preloaded Mouse Data'),
                                dcc.Dropdown(
                                    id='mouse-selector',
                                    options={
                                        'visual': 'Visual Cortex',
                                        'motor': 'Motor Cortex'
                                    },
                                    value='visual'
                                ),

                            ]
                        ),

                        html.Div(
                            id='operations-block',
                            children=[
                                html.H3('Operations', className='block-title'),
                                html.Div([
                                    html.Div([
                                        html.Button('Align Datasets', id='btn-align'),
                                        dcc.RadioItems(
                                            id='eig-method',
                                            options={
                                                'eig': 'eig',
                                                'eigs': 'eigs'
                                            },
                                            value='eig'
                                        ),
                                        html.Button('Identify Clusters', id='btn-cluster'),

                                    ], style={'flex': 1}),
                                    html.Div([
                                        dcc.Loading(id='loading-1', children=html.Div(id='loading-output-1')),
                                        html.Br(),
                                        dcc.Loading(id='loading-2', children=html.Div(id='loading-output-2')),
                                    ], style={'flex': 1})
                                ], style={'display': 'flex'})
                            ]
                        ),

                        html.Div(
                            id='controls-block',
                            children=[
                                html.H3('Plot Controls', className='block-title'),
                                html.Div([
                                    html.H4('Plot type'),
                                    dcc.Dropdown(
                                        id='plot-type',
                                        options={
                                            'alignment': 'Dataset alignment',
                                            'separate2': 'Separate 2-D plots',
                                            'separate3': 'Separate 3-D plots',
                                            'bibiplot': 'Bibiplot'
                                        }
                                    ),

                                    html.H4('Color by...'),
                                    dcc.RadioItems(
                                        id='color-type',
                                        options=color_types,
                                        value='ttype'
                                    ),
                                ])
                            ]
                        )
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
                                    figure=go.Figure(),
                                    style={'height': '600px', 'width': '100%'}
                                ),
                            ]
                        )

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
