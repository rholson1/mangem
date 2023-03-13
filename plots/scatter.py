from plotly.subplots import make_subplots
import plotly.express as px
from operator import itemgetter
from app_main.constants import color_types, marker_size, font_size



def scatter2d(df1, df2, x, y, color_type, metadata_type, label_1, label_2, size_key='default'):
    """
    Plot two 2D plots side by side
    :param df1: Pandas dataframe 1
    :param df2: Pandas dataframe 2
    :param x: column number for x-coordinate
    :param y: column number for y-coordinate
    :param color_type: column name for color
    :param metadata_type: name of metadata column to be used for color if color_type == 'metadata'
    :param label_1: label for df1
    :param label_2: label for df2
    :return:
    """

    # Get the column names from the 0-based index
    x_name = df1.columns[x]
    y_name = df1.columns[y]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(label_1, label_2))

    color_col = metadata_type if color_type == 'metadata' else color_type

    if color_col:
        fig_e = px.scatter(df1, x=x_name, y=y_name, color=color_col, color_continuous_scale=px.colors.sequential.Blackbody)
        fig_g = px.scatter(df2, x=x_name, y=y_name, color=color_col, color_continuous_scale=px.colors.sequential.Blackbody)

        # Suppress legend for one of the plots to avoid double entries
        for d in fig_e.data:
            d['showlegend'] = False

        # sort fig_*.data to get a sorted legend
        fig.add_traces(sorted(fig_e.data, key=itemgetter('legendgroup')), rows=1, cols=1)
        fig.add_traces(sorted(fig_g.data, key=itemgetter('legendgroup')), rows=1, cols=2)

        # Set the legend title
        fig.update_layout(legend_title=metadata_type if color_type == 'metadata' else color_types[color_type],
                          legend={'itemsizing': 'constant'})

    else:
        # Uncolored plots
        fig_e = px.scatter(df1, x=x_name, y=y_name)
        fig_g = px.scatter(df2, x=x_name, y=y_name)

        fig.add_trace(fig_e.data[0], row=1, col=1)
        fig.add_trace(fig_g.data[0], row=1, col=2)

    # This appears to link x and y axes so both plots have same level of pan/zoom
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    plot_font_size = font_size[size_key]['plot_font_size']
    plot_title_font_size = font_size[size_key]['plot_title_font_size']

    fig.update_layout(title_text=f'Aligned Cells (2D)',
                      font_size=plot_font_size,
                      title_font_size=plot_title_font_size,
                      title_yanchor='bottom',
                      title_pad={'b': plot_title_font_size * 1.5},
                      margin={'t': 100 + plot_title_font_size * 1.5}
                      )
    fig.update_annotations(font_size=plot_title_font_size)  # subplot titles are annotations

    return fig


def scatter3d(df1, df2, x, y, z, color_type, metadata_type, relayoutData, label_1, label_2, size_key='default'):
    """
    Plot two 2D plots side by side
    :param df1: Pandas dataframe 1
    :param df2: Pandas dataframe 2
    :param x: column number for x-coordinate
    :param y: column number for y-coordinate
    :param z: column number for z-coordinate
    :param color_type: column name for color
    :param metadata_type: name of metadata column to be used for color if color_type == 'metadata'
    :param relayoutData: relayoutData from figure
    :param label_1: label for df1
    :param label_2: label for df2
    :return:
    """

    # Get the column names from the 0-based index
    x_name = df1.columns[x]
    y_name = df1.columns[y]
    z_name = df1.columns[z]

    marker_size_3d = marker_size[size_key]['3d']
    tickfont_size = font_size[size_key]['tickfont_size']

    fig = make_subplots(rows=1, cols=2, subplot_titles=(label_1, label_2),
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    color_col = metadata_type if color_type == 'metadata' else color_type

    if color_col:
        fig_e = px.scatter_3d(df1, x=x_name, y=y_name, z=z_name,
                              color=color_col, color_continuous_scale=px.colors.sequential.Blackbody).update_traces(
                    marker={'size': marker_size_3d})
        fig_g = px.scatter_3d(df2, x=x_name, y=y_name, z=z_name,
                              color=color_col, color_continuous_scale=px.colors.sequential.Blackbody).update_traces(
                    marker={'size': marker_size_3d})

        # Suppress legend for one of the plots to avoid double entries
        for d in fig_e.data:
            d['showlegend'] = False

        # sort fig_*.data to get a sorted legend
        fig.add_traces(sorted(fig_e.data, key=itemgetter('legendgroup')), rows=1, cols=1)
        fig.add_traces(sorted(fig_g.data, key=itemgetter('legendgroup')), rows=1, cols=2)

        # Set the legend title
        fig.update_layout(legend_title=metadata_type if color_type == 'metadata' else color_types[color_type],
                          legend={'itemsizing': 'constant'})

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
    else:
        # default view, zoomed out so axes are fully visible
        camera = {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}}
    if camera:
        fig.update_layout(scene_camera=camera, scene2_camera=camera)

    plot_font_size = font_size[size_key]['plot_font_size']
    plot_title_font_size = font_size[size_key]['plot_title_font_size']

    fig.update_layout(title_text=f'Aligned Cells (3D)',
                      font_size=plot_font_size,
                      title_font_size=plot_title_font_size,
                      title_yanchor='bottom',
                      title_pad={'b': plot_title_font_size * 1.5},
                      margin={'t': 100 + plot_title_font_size * 1.5})
    fig.update_annotations(font_size=plot_title_font_size)  # subplot titles are annotations
    fig.update_scenes(xaxis_tickfont_size=tickfont_size,
                      yaxis_tickfont_size=tickfont_size,
                      zaxis_tickfont_size=tickfont_size)
    return fig
