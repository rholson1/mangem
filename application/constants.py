from enum import Enum


color_types = {'cluster': 'Cluster', 'ttype': 't-type'}

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
