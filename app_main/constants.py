from enum import Enum

marker_size_3d = 1.25  #0.75

big_fonts = False
if big_fonts:
    plot_font_size = 16  # default 12
    plot_title_font_size = 22  # default 16
else:
    plot_font_size = 12
    plot_title_font_size = 16

#color_types = {'cluster': 'Cluster', 'ttype': 't-type'}
color_types = {'cluster': 'Cross-modal Cluster', 'metadata': 'Metadata'}

dataset_titles = {
    'motor': 'Mouse Motor Cortex',
    'visual': 'Mouse Visual Cortex',
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
