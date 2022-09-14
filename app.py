from dash import Dash, html, dcc, Output, Input, State, MATCH
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from operator import itemgetter
import time

app = Dash(__name__)
app.title = 'Multimodal Alignment'
server = app.server


color_types = {'ttype': 't-type', 'gmm_cluster': 'Cluster'}


# read aligned data files.
efeatures_NMA = pd.read_csv('data/mouse_visual_cortex/efeature_NMA.csv')
Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

geneExp_NMA = pd.read_csv('data/mouse_visual_cortex/geneExp_NMA.csv')
Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

# plot both sets of data (first three dimensions)
fig_e = px.scatter_3d(Xe, x=0, y=1, z=2).update_traces(name='Electrophys', marker={'color': 'red', 'size': 0.75}, showlegend=True)
fig_g = px.scatter_3d(Xg, x=0, y=1, z=2).update_traces(name='Gene Expression', marker={'color': 'blue', 'size': 0.75}, showlegend=True)

# fig_e_trace = fig_e.data[0]
# fig_e_trace['name'] = 'Electrophys'
#
# fig_g_trace = fig_g.data[0]
# fig_g_trace['name'] = 'Gene Expression'


fig_combined = go.Figure(data=fig_e.data + fig_g.data)

fig_combined.update_layout(
    title='Combined Plot in Latent Space',
    showlegend=True,
    scene={
        'xaxis_title': '',
        'yaxis_title': '',
        'zaxis_title': '',
    }
    # scene={
    #     'xaxis_title': 'Component 0',
    #     'yaxis_title': 'Component 1',
    #     'zaxis_title': 'Compenent 2',
    # }
    # ,
    # legend_title='the legend'
)

app.layout = html.Div(children=[
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
                                    html.Button('Identify Clusters', id='btn-cluster')
                                ], style={'flex':1}),
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
                                figure=fig_combined,
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


# Pattern-matching callback allows one function to handle all the file uploads.
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
    Input(component_id='mouse-selector', component_property='value'),
    Input(component_id='plot-type', component_property='value'),
    Input(component_id='color-type', component_property='value'),
)
def update_alignment_plot(dataset, plot_type, color_type):
    # read aligned data files.
    if not dataset:
        return go.Figure()

    efeatures_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/efeature_NMA.csv')
    efeatures_NMA['gmm_cluster'] = efeatures_NMA['gmm_cluster'].astype('string')
    Xe = np.array(efeatures_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

    geneExp_NMA = pd.read_csv(f'data/mouse_{dataset}_cortex/geneExp_NMA.csv')
    geneExp_NMA['gmm_cluster'] = geneExp_NMA['gmm_cluster'].astype('string')
    Xg = np.array(geneExp_NMA.drop(columns=['data', 'ttype', 'gmm_cluster', 'cellnames']))

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
            legend = {'itemsizing': 'constant'}
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
            fig_e = px.scatter_3d(efeatures_NMA, x='Val0', y='Val1', z='Val2', color=color_type).update_traces(marker={'size': 0.75})
            fig_g = px.scatter_3d(geneExp_NMA, x='Val0', y='Val1', z='Val2', color=color_type).update_traces(marker={'size': 0.75})

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
    else:
        # No plot type specified - show a blank plot
        fig = go.Figure().update_layout()

    return fig


@app.callback(
    Output(component_id='loading-output-1', component_property='children'),
    Input(component_id='btn-align', component_property='n_clicks'),
    prevent_initial_call=True
)
def align_datasets(value):
    # perform dataset alignment
    time.sleep(2)
    return 'Dataset aligned'

@app.callback(
    Output(component_id='loading-output-2', component_property='children'),
    Input(component_id='btn-cluster', component_property='n_clicks'),
    prevent_initial_call=True
)
def identify_clusters(value):
    # perform dataset alignment
    time.sleep(2)
    return 'Clusters identified'


if __name__ == '__main__':
    app.run_server(debug=True)