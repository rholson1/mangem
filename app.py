from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = Dash(__name__)
server = app.server


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

upload_style = {
    'width': '60%',
    'height': '10px',
    'lineHeight': '15px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px',
    'padding': '20px',
    'display': 'inline-block'

}

app.layout = html.Div(children=[
    html.H1(children='BRAIN - Multimodal Alignment'),

    html.Div(children='''
        Demonstration of plotting aligned data in latent space (first three dimensions).
    '''),

    html.Div(children=[
        html.Div(id='controls_container', children=[
            html.P("[Inactive controls]"),
            dcc.Upload(
                id='upload_1',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select File 1')
                ]),
                style=upload_style,
                # Allow multiple files to be uploaded
                multiple=False
            ),
            dcc.Input(
                id='upload_1_label',
                type='text',
                placeholder='File 1 Label'
            ),

            dcc.Upload(
                id='upload_2',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select File 2')
                ]),
                style=upload_style,
                # Allow multiple files to be uploaded
                multiple=False
            ),
            dcc.Input(
                id='upload_2_label',
                type='text',
                placeholder='File 2 Label'
            ),

        ], style={'flex': 1}),
        html.Div(id='graph_container', children=[
            dcc.Graph(
                id='graph_combined',
                figure=fig_combined,
                style={'height': '600px', 'width': '100%'}
            ),
        ], style={'flex': 1})
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'border': '1.5px solid gray',
        'border-radius': '5px',
        'padding': '10px'
    })
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

if __name__ == '__main__':
    app.run_server(debug=True)