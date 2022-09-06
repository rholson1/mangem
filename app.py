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
fig_e = px.scatter_3d(Xe, x=0, y=1, z=2).update_traces(marker={'color': 'red', 'size': 0.5})
fig_g = px.scatter_3d(Xg, x=0, y=1, z=2).update_traces(marker={'color': 'blue', 'size': 0.5})

fig_combined = go.Figure(data=fig_e.data + fig_g.data)


app.layout = html.Div(children=[
    html.H1(children='BRAIN - Multimodal Alignment'),

    html.Div(children='''
        Demonstration of plotting aligned data in latent space (first three dimensions).
    '''),

    dcc.Graph(
        id='graph_combined',
        figure=fig_combined,
    ),

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