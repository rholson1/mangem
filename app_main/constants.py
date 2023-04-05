from enum import Enum

#marker_size_3d = 1.25  #0.75

marker_size = {
    'big': {
        '3d': 2,
        '2d': 6
    },
    'default': {
        '3d': 1.25,
        '2d': 3
    }
}

font_size = {
    'big': {
        'tickfont_size_3d': 20,
        'tickfont_size': 26,
        'plot_font_size': 44,  # 32,  # 20
        'bibiplot_long_font_size': 30,
        'plot_title_font_size': 60,  # 44  # 32
    },
    'default': {
        'tickfont_size_3d': 12,
        'tickfont_size': 12,
        'plot_font_size': 12,
        'bibiplot_long_font_size': 12,
        'plot_title_font_size': 16
    }
}

plot_size = {
    'big': {'height': '1200px', 'width': '2000px'},
    'default': {'height': '600px', 'width': '1000px'}
}

# big_fonts = False
# if big_fonts:
#     plot_font_size = 20  # default 12
#     plot_title_font_size = 32  # default 16
# else:
#     plot_font_size = 12
#     plot_title_font_size = 16
#
# big_plots = False
# if big_plots:
#     plot_size_style = {'height': '1200px', 'width': '2000px'}
# else:
#     plot_size_style = {'height': '600px', 'width': '1000px'}

#color_types = {'cluster': 'Cluster', 'ttype': 't-type'}
color_types = {'cluster': 'Cross-modal Cluster', 'metadata': 'Metadata'}
color_types_title = {'cluster': 'Cross-modal<br>Cluster', 'metadata': 'Metadata'}

dataset_titles = {
    'motor': 'Mouse Motor Cortex: gene exp. & electrophys.',
    'visual': 'Mouse Visual Cortex: gene exp. & electrophys.',
    'morph': 'Mouse Visual Cortex: gene exp. & morphology',
    'upload': 'Custom Data'
    }

blank_layout = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'xaxis': {'visible': False, 'showline': False, 'showgrid': False},
    'yaxis': {'visible': False, 'showline': False, 'showgrid': False},
}

class UploadFileType(int, Enum):
    DATA_1 = 1
    DATA_2 = 2
    METADATA = 3

class StoredFileType(int, Enum):
    ALIGNED_1 = 1
    ALIGNED_2 = 2
    STATUS = 3
