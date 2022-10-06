
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np
import scipy.spatial.distance as sd
from operations.alignment import calc_domainAveraged_FOSCTTM



def create_alignment_plot(Xe, Xg, dataset, x, y, z):


    pairwise_distances = [sd.euclidean(Xe[i, :], Xg[i, :]) for i in range(Xe.shape[0])]
    foscttm = calc_domainAveraged_FOSCTTM(Xe, Xg)

    titles = (f'$\mu = {np.mean(pairwise_distances):.4f}$', f'$\mu = {np.mean(foscttm):.4f}$', 'Alignment')

    fig = make_subplots(rows=1, cols=3,
                        column_widths=[0.2, 0.2, 0.6],
                        column_titles=titles,
                        specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'scene'}]])




    # create box plot of distances between corresponding cells in latent space
    fig.add_trace(go.Box(y=pairwise_distances, name='Pairwise cell distance', showlegend=False), row=1, col=1)

    # create box plot of FOSCTTM scores
    fig.add_trace(go.Box(y=foscttm, name='FOSCTTM', showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter3d(x=Xe[:, x], y=Xe[:, y], z=Xe[:, z],
                               mode='markers',
                               marker={'color': 'red', 'size': 0.75},
                               name='Electrophys',
                               showlegend=True),
                  row=1, col=3)
    fig.add_trace(go.Scatter3d(x=Xg[:, x], y=Xg[:, y], z=Xg[:, z],
                               mode='markers',
                               marker={'color': 'blue', 'size': 0.75},
                               name='Gene Expression',
                               showlegend=True),
                  row=1, col=3)


    fig.update_layout(title_text=f'Dataset alignment in latent space: Mouse {dataset.title()} Cortex',
                      legend={'itemsizing': 'constant'})

    return fig


def plot_alignment(Xe, Xg, dataset, x, y, z):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=Xe[:, x], y=Xe[:, y], z=Xe[:, z],
                               mode='markers',
                               marker={'color': 'red', 'size': 0.75},
                               name='Electrophys',
                               showlegend=True))
    fig.add_trace(go.Scatter3d(x=Xg[:, x], y=Xg[:, y], z=Xg[:, z],
                               mode='markers',
                               marker={'color': 'blue', 'size': 0.75},
                               name='Gene Expression',
                               showlegend=True))
    fig.update_layout(title_text=f'Dataset alignment in latent space: Mouse {dataset.title()} Cortex',
                      legend={'itemsizing': 'constant'})

    return fig

def plot_alignment_error(Xe, Xg, dataset):

    pairwise_distances = [sd.euclidean(Xe[i, :], Xg[i, :]) for i in range(Xe.shape[0])]
    foscttm = calc_domainAveraged_FOSCTTM(Xe, Xg)

    titles = (f'$\mu = {np.mean(pairwise_distances):.4f}$', f'$\mu = {np.mean(foscttm):.4f}$')

    fig = make_subplots(rows=1, cols=2,
                        column_titles=titles,
                        specs=[[{'type': 'xy'}, {'type': 'xy'}]])

    # create box plot of distances between corresponding cells in latent space
    fig.add_trace(go.Box(y=pairwise_distances, name='Pairwise cell distance', showlegend=False), row=1, col=1)

    # create box plot of FOSCTTM scores
    fig.add_trace(go.Box(y=foscttm, name='FOSCTTM', showlegend=False), row=1, col=2)

    fig.update_layout(title_text=f'Dataset alignment in latent space: Mouse {dataset.title()} Cortex',
                      legend={'itemsizing': 'constant'})

    return fig