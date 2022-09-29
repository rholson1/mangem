
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import scipy.spatial.distance as sd



def create_alignment_plot(Xe, Xg, dataset):
    titles = ('Pairwise cell distance', 'Alignment')

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.3, 0.7],
                        column_titles=titles,
                        specs=[[{'type': 'xy'}, {'type': 'scene'}]])

    # plot both sets of data (first three dimensions)
    # fig_e = px.scatter_3d(Xe, x=0, y=1, z=2).update_traces(name='Electrophys',
    #                                                        marker={'color': 'red', 'size': 0.75},
    #                                                        showlegend=True)

    fig.add_trace(go.Scatter3d(x=Xe[:,0], y=Xe[:,1], z=Xe[:,2],
                               mode='markers',
                               marker={'color': 'red', 'size': 0.75},
                               name='Electrophys',
                               showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter3d(x=Xg[:, 0], y=Xg[:, 1], z=Xg[:, 2],
                               mode='markers',
                               marker={'color': 'blue', 'size': 0.75},
                               name='Gene Expression',
                               showlegend=True),
                  row=1, col=2)

    # fig_g = px.scatter_3d(Xg, x=0, y=1, z=2).update_traces(name='Gene Expression',
    #                                                        marker={'color': 'blue', 'size': 0.75},
    #                                                        showlegend=True)
    # fig1 = go.Figure(data=fig_e.data + fig_g.data)
    #
    # fig1.update_layout(
    #     title=f'Multimodal Alignment in Latent Space: Mouse {dataset.title()} Cortex',
    #     showlegend=True,
    #     scene={
    #         'xaxis_title': '',
    #         'yaxis_title': '',
    #         'zaxis_title': '',
    #     },
    #     legend={'itemsizing': 'constant'}
    # )


    # create box plot of distances between corresponding cells in latent space
    distances = [sd.euclidean(Xe[i, :], Xg[i, :]) for i in range(Xe.shape[0])]

    # fig2 = go.Figure()
    # fig2.add_trace(go.Box(y=distances, name='Pairwise cell distance', showlegend=False))
    # fig2.update_layout(title='Pairwise cell distance in latent space after alignment')

    fig.add_trace(go.Box(y=distances, name='Pairwise cell distance', showlegend=False), row=1, col=1)
    fig.update_layout(title_text=f'Dataset alignment in latent space: Mouse {dataset.title()} Cortex')


    return fig
