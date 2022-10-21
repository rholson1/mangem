from plotly.subplots import make_subplots
import plotly.express as px
from operator import itemgetter
from application.settings import color_types


def scatter2d(df1, df2, x, y, color_type):
    """
    Plot two 2D plots side by side
    :param df1: Pandas dataframe 1
    :param df2: Pandas dataframe 2
    :param x: column number for x-coordinate
    :param y: column number for y-coordinate
    :param color: column name for color
    :return:
    """

    # Get the column names from the 0-based index
    x_name = df1.columns[x]
    y_name = df1.columns[y]

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Electrophys', 'Gene Expression'))

    if color_type:
        fig_e = px.scatter(df1, x=x_name, y=y_name, color=color_type)
        fig_g = px.scatter(df2, x=x_name, y=y_name, color=color_type)

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
        fig_e = px.scatter(df1, x=x_name, y=y_name)
        fig_g = px.scatter(df2, x=x_name, y=y_name)

        fig.add_trace(fig_e.data[0], row=1, col=1)
        fig.add_trace(fig_g.data[0], row=1, col=2)

    # This appears to link x and y axes so both plots have same level of pan/zoom
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig.update_layout(title_text=f'2D projections in latent space')

    return fig


def scatter3d(df1, df2, x, y, z, color_type, relayoutData):
    """
    Plot two 2D plots side by side
    :param df1: Pandas dataframe 1
    :param df2: Pandas dataframe 2
    :param x: column number for x-coordinate
    :param y: column number for y-coordinate
    :param z: column number for z-coordinate
    :param color_type: column name for color
    :param relayoutData: relayoutData from figure
    :return:
    """

    # Get the column names from the 0-based index
    x_name = df1.columns[x]
    y_name = df1.columns[y]
    z_name = df1.columns[z]

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Electrophys', 'Gene Expression'),
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    if color_type:
        fig_e = px.scatter_3d(df1, x=x_name, y=y_name, z=z_name, color=color_type).update_traces(
                    marker={'size': 0.75})
        fig_g = px.scatter_3d(df2, x=x_name, y=y_name, z=z_name, color=color_type).update_traces(
                    marker={'size': 0.75})

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
        fig_e = px.scatter_3d(df1, x=x_name, y=y_name, z=z_name)
        fig_g = px.scatter_3d(df2, x=x_name, y=y_name, z=z_name)

        fig.add_trace(fig_e.data[0], row=1, col=1)
        fig.add_trace(fig_g.data[0], row=1, col=2)

    # synchronize zoom/rotation of plots
    camera = None
    if 'scene.camera' in relayoutData:
        camera = relayoutData['scene.camera']
    elif 'scene2.camera' in relayoutData:
        camera = relayoutData['scene2.camera']
    if camera:
        fig.update_layout(scene_camera=camera, scene2_camera=camera)

    fig.update_layout(title_text=f'3D projections in latent space')

    return fig