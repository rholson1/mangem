from dash import html, dcc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html
import uuid

from app_main.settings import appname
from app_main.constants import UploadFileType, blank_layout, color_types
from app_main.utilities import unique_visitors


def get_layout():
    session_id = str(uuid.uuid4())

    layout = html.Div(children=[
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session_id', data=session_id),
        dcc.Store(id='store-data_selected', data='0'),
        dcc.Store(id='store-aligned', data='0'),
        dcc.Store(id='store-align-bg-trigger', data='0'),
        dcc.Store(id='store-label-1', data='Modality 1'),
        dcc.Store(id='store-label-2', data='Modality 2'),
        dcc.Store(id='store-bg-vars', data=''),
        dcc.Store(id={'type': 'store-upload', 'index': UploadFileType.DATA_1}, data='0'),
        dcc.Store(id={'type': 'store-upload', 'index': UploadFileType.DATA_2}, data='0'),
        dcc.Store(id={'type': 'store-upload', 'index': UploadFileType.METADATA}, data='0'),

        html.Div(id='id_header',
                 children=[
                     html.Div(id='id_topbar',
                              children=[
                                  html.Div(html.A('UNIVERSITY of WISCONSIN-MADISON', href='http://www.wisc.edu')),
                                  html.Div(html.A('DAIFENG WANG LAB', href='http://daifengwanglab.org'))
                              ]),
                     html.Div(id='id_titlebar',
                              children=[
                                  html.H1(children=appname)
                              ])

                 ]),

        html.Div(
            id='main-block',
            children=[
                html.Div(
                    id='left-panel',
                    children=[
                        dcc.Tabs(
                            id='left-panel-tabs',
                            className='custom-tabs-container',
                            children=[
                                dcc.Tab(
                                    label='About',
                                    value='tab-1',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    children=[
                                        html.H3(f'{appname}'),
                                        'Recent single-cell multimodal data reveal multi-scale characteristics '
                                        'of single cells, such as transcriptomics, morphology, and '
                                        'electrophysiology. However, integrating and analyzing such multimodal '
                                        'data to deeper understand functional genomics and gene regulation in '
                                        'various cellular characteristics remains elusive. This web app implements '
                                        'multiple machine learning methods for single-cell multi-modal data integration '
                                        'and analysis, including emerging manifold alignment methods. The users can '
                                        'upload their multi-modal data and select the alignment method. The aligned '
                                        'cells from multimodalities can be clustered into multimodal cell clusters. '
                                        'Further cluster analyses can reveal corresponding features across modalities. '
                                        'The technical details including pre-loaded datasets are available in our '
                                        'publication, ',
                                        html.A(href='https://doi.org/10.1038/s42003-021-02807-6',
                                               children=(#'Huang, J., Sheng, J. & Wang, D. Manifold learning analysis suggests strategies to align single-cell multimodal data of neuronal electrophysiology and transcriptomics. Commun Biol 4, 1308 (2021). https://doi.org/10.1038/s42003-021-02807-6'
                                                         'Manifold learning analysis suggests strategies for aligning '
                                                         'single-cell multi-modalities and revealing functional genomics '
                                                         'for neuronal electrophysiology'
                                                         ),
                                               target='_blank'),
                                        '.',
                                        html.P([
                                            'For questions, contact ',
                                            html.A('Daifeng Wang, Ph.D.', href='mailto:daifeng.wang@wisc.edu'),
                                        ]),
                                        html.P([
                                            html.A(href='https://mediaspace.wisc.edu/media/MANGEM+Demo/1_99t20udb',
                                                   children=['Video Demo'], target='_blank'),
                                            '']
                                        ),
                                        html.H3('App compatibility'),
                                        html.P('This app has been tested on the following operating systems and browsers'),
                                        html.Table(id='compat-table', children=[
                                            html.Thead(
                                                html.Tr([
                                                    html.Th('OS'),
                                                    html.Th('Chrome'),
                                                    html.Th('Edge'),
                                                    html.Th('Firefox'),
                                                    html.Th('Safari')
                                                ]),
                                            ),
                                            html.Tbody([
                                                html.Tr([
                                                    html.Th('Linux'),
                                                    html.Td('x'),
                                                    html.Td(''),
                                                    html.Td('x'),
                                                    html.Td('')
                                                ]),
                                                html.Tr([
                                                    html.Th('MacOS'),
                                                    html.Td('x'),
                                                    html.Td(''),
                                                    html.Td(''),
                                                    html.Td('x')
                                                ]),
                                                html.Tr([
                                                    html.Th('Windows'),
                                                    html.Td('x'),
                                                    html.Td('x'),
                                                    html.Td('x'),
                                                    html.Td('')
                                                ]),
                                            ])
                                        ]),
                                        html.Br(),
                                        html.P(children=unique_visitors())
                                    ]
                                ),

                                dcc.Tab(
                                    label='Step 1 \nUpload Data',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    # id='data-block',
                                    children=[
                                        html.Div(id='data-block', children=[
                                            html.Div(children=[
                                                html.H3('Data', className='block-title'),
                                                html.H4([
                                                    'Select Data',
                                                    html.I(id='data-select-tooltip',
                                                           className='fa-regular fa-circle-question')
                                                ]),
                                                dbc.Tooltip(
                                                    [
                                                        'Select a preloaded dataset or upload your own .csv files.',
                                                        html.H3('Preloaded Datasets'),
                                                        html.Ol([
                                                            'Two main data sources are utilized for MANGEM’s pre-packaged datasets.  First, patch-seq data of 3,654 single-cells from the ',
                                                            html.A('mouse visual cortex', href='https://pubmed.ncbi.nlm.nih.gov/33186530/'),
                                                            ' was used to generate ',
                                                            html.A('electrophysiological and morphological features', href='https://www.nature.com/articles/s42003-021-02807-6'),
                                                            '.  MANGEM provides a preset combining gene expression (1,302 genes) and electrophysiological features (41 features) on this dataset.  ',
                                                            'Second, 1,208 single-cells from the ',
                                                            html.A('mouse motor cortex', href='https://www.nature.com/articles/s41586-020-2907-3'),
                                                            ' were processed similarly (1,286 genes, 29 features).  MANGEM provides two presets from this dataset, each containing gene expression ',
                                                            'and one of electrophysiological or morphological features.  The latter preset includes a subset of 646 cells with 1000 genes and 61 morphological features.'
                                                        ])

                                                     ],
                                                    target='data-select-tooltip', trigger='legacy'
                                                ),
                                                dcc.Dropdown(
                                                    id='data-selector',
                                                    options=[
                                                        {'label': 'Mouse Motor Cortex: gene expression/electrohphysiology (1208 cells)', 'value': 'motor'},
                                                        {'label': 'Mouse Visual Cortex: gene expression/electrophysiology (3654 cells)', 'value': 'visual'},
                                                        {'label': 'Mouse Motor Cortex: gene expression/morphology (646 cells)', 'value': 'morph'},
                                                        {'label': 'Upload your data!', 'value': 'upload'},
                                                    ],
                                                    
                                                    
                                                    # {
                                                    #     'motor': 'Mouse Motor Cortex (1208 cells)',
                                                    #     'visual': 'Mouse Visual Cortex (3654 cells)',
                                                    #     'upload': 'Upload your data!'
                                                    # },
                                                    value=''
                                                ),

                                                html.Div(
                                                    id='upload-container',
                                                    className='hidden',
                                                    children=[

                                                        html.Div(
                                                            className='upload-block',
                                                            children=[
                                                                html.H4([
                                                                    'Modality 1',
                                                                    html.I(id='modality-1-tooltip',
                                                                           className='fa-regular fa-circle-question')
                                                                ]),
                                                                dbc.Tooltip(children=[
                                                                    'Each uploaded .csv file should have the same number '
                                                                    'of rows and should have a consistent cell (row) order. ',
                                                                    html.Table(className='table-csv', children=[
                                                                        html.Tr([
                                                                            html.Td(),
                                                                            html.Td('Feat. 1'),
                                                                            html.Td('Feat. 2'),
                                                                            html.Td('Feat. 3'),
                                                                            html.Td('...'),]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 1'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td(''),]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 2'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td(''),]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 3'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('')]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('...'),
                                                                            html.Td(''),
                                                                            html.Td(''),
                                                                            html.Td(''),
                                                                            html.Td(''),]
                                                                        )]
                                                                    )


                                                                    ],
                                                                    target='modality-1-tooltip',
                                                                    trigger='legacy'
                                                                ),
                                                                dcc.Input(
                                                                    id='upload_1_label',
                                                                    type='text',
                                                                    value='',
                                                                    placeholder='Modality 1 Label'
                                                                ),
                                                                dcc.Upload(id={'type': 'dynamic-upload',
                                                                               'index': UploadFileType.DATA_1},
                                                                           children=html.Button('Upload Data'),
                                                                           className='upload'),
                                                                html.Div(id={'type': 'dynamic-upload-info',
                                                                             'index': UploadFileType.DATA_1}),
                                                                html.Div(id={'type': 'dynamic-output',
                                                                             'index': UploadFileType.DATA_1}),

                                                            ]
                                                        ),
                                                        html.Div(
                                                            className='upload-block',
                                                            children=[
                                                                html.H4([
                                                                    'Modality 2',
                                                                    html.I(id='modality-2-tooltip',
                                                                           className='fa-regular fa-circle-question')
                                                                ]),
                                                                dbc.Tooltip(children=[
                                                                    'Each uploaded .csv file should have the same number '
                                                                    'of rows and should have a consistent cell (row) order. ',
                                                                    html.Table(className='table-csv', children=[
                                                                        html.Tr([
                                                                            html.Td(),
                                                                            html.Td('Feat. 1'),
                                                                            html.Td('Feat. 2'),
                                                                            html.Td('Feat. 3'),
                                                                            html.Td('...'),]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 1'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td(''),]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 2'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td(''),]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 3'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('#'),
                                                                            html.Td('')]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('...'),
                                                                            html.Td(''),
                                                                            html.Td(''),
                                                                            html.Td(''),
                                                                            html.Td(''),]
                                                                        )]
                                                                    )
                                                                    ],
                                                                    target='modality-2-tooltip',
                                                                    trigger='legacy'
                                                                ),
                                                                dcc.Input(
                                                                    id='upload_2_label',
                                                                    type='text',
                                                                    value='',
                                                                    placeholder='Modality 2 Label'
                                                                ),
                                                                dcc.Upload(id={'type': 'dynamic-upload',
                                                                               'index': UploadFileType.DATA_2},
                                                                           children=html.Button('Upload Data'),
                                                                           className='upload'),
                                                                html.Div(id={'type': 'dynamic-upload-info',
                                                                             'index': UploadFileType.DATA_2}),
                                                                html.Div(id={'type': 'dynamic-output',
                                                                             'index': UploadFileType.DATA_2}),
                                                            ]
                                                        ),
                                                        html.Div(
                                                            className='upload-block',
                                                            children=[
                                                                html.H4([
                                                                    'Metadata',
                                                                    html.I(id='metadata-tooltip',
                                                                           className='fa-regular fa-circle-question')
                                                                ]),
                                                                dbc.Tooltip(children=[
                                                                    'Additional cell phenotype information can be supplied in a metadata .csv file.',
                                                                    html.Table(className='table-csv', children=[
                                                                        html.Tr([
                                                                            html.Td(),
                                                                            html.Td('Phenotype 1'),
                                                                            html.Td('...'), ]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 1'),
                                                                            html.Td('#'),
                                                                            html.Td(''), ]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 2'),
                                                                            html.Td('#'),
                                                                            html.Td(''), ]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('Cell 3'),
                                                                            html.Td('#'),
                                                                            html.Td('')]
                                                                        ),
                                                                        html.Tr([
                                                                            html.Td('...'),
                                                                            html.Td(''),
                                                                            html.Td(''), ]
                                                                        )]
                                                                               )
                                                                    ],
                                                                    target='metadata-tooltip',
                                                                    trigger='legacy'
                                                                ),
                                                                dcc.Upload(id={'type': 'dynamic-upload',
                                                                               'index': UploadFileType.METADATA},
                                                                           children=html.Button('Upload Metadata'),
                                                                           className='upload'),
                                                                html.Div(id={'type': 'dynamic-upload-info',
                                                                             'index': UploadFileType.METADATA}),
                                                                html.Div(id={'type': 'dynamic-output',
                                                                             'index': UploadFileType.METADATA})
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                html.H4([
                                                    'Preprocessing',
                                                    html.I(id='preprocessing-tooltip',
                                                           className='fa-regular fa-circle-question')
                                                ]),
                                                dbc.Tooltip(
                                                    'Each dataset may optionally be transformed prior to alignment.',
                                                    target='preprocessing-tooltip', trigger='legacy'
                                                ),
                                                html.Label([
                                                    html.Span('', id='preprocess-label-1'),
                                                    ': ',
                                                    dcc.Dropdown(
                                                        id='preprocess_1',
                                                        options={
                                                            'none': 'None',
                                                            'scale': 'Standardize',
                                                            'log': 'Log transform'
                                                        },
                                                        value='log'
                                                    )
                                                ]),
                                                html.Label([
                                                    html.Span('', id='preprocess-label-2'),
                                                    ': ',
                                                    dcc.Dropdown(
                                                        id='preprocess_2',
                                                        options={
                                                            'none': 'None',
                                                            'scale': 'Standardize',
                                                            'log': 'Log transform'
                                                        },
                                                        value='scale'
                                                    )
                                                ]),
                                                html.H4([
                                                    'Explore data',
                                                    html.I(id='explore-tooltip',
                                                           className='fa-regular fa-circle-question')
                                                ]),
                                                dbc.Tooltip([
                                                    'Enter one or two variables (for example, genes) in the search box '
                                                    'and select a cell phenotype from the metadata dropdown '
                                                    'to see a set of boxplots (one variable) or a scatter plot '
                                                    '(two variables).'
                                                ], target='explore-tooltip', trigger='legacy'),
                                                html.Label([
                                                    html.Span('', id='explore-label-1'),
                                                    ': ',
                                                    html.Br(),
                                                    dcc.Dropdown(id='explore-var1', placeholder='Feature 1'),
                                                    html.Button(id='use-explore-var1', children='+'),
                                                ]),
                                                html.Label([
                                                    html.Span('', id='explore-label-2'),
                                                    ': ',
                                                    html.Br(),
                                                    dcc.Dropdown(id='explore-var2', placeholder='Feature 2'),
                                                    html.Button(id='use-explore-var2', children='+')
                                                ]),
                                                html.Label([
                                                    'X, Y: ',
                                                    dcc.Input(id='explore-vars',
                                                              type='text',
                                                              debounce=True,
                                                              placeholder='X1 or X1,X2'),
                                                    dcc.Checklist(id='explore-preprocess',
                                                                  options=[{'label': 'Preprocess', 'value': 'preprocess'}]),

                                                    html.Label([
                                                        'Use log axis:',
                                                        dcc.Checklist(id='explore-log-axis',
                                                                      options=['X', 'Y'], inline=True),
                                                    ], id='explore-log-axis-label'),

                                                ]),
                                                html.Label([
                                                    'Metadata: ',
                                                    dcc.Dropdown(
                                                        id='metadata-type-x',
                                                        options={
                                                            'ttype': 'Transcriptomic Type'
                                                        },
                                                        placeholder='Select metadata...',
                                                    ),
                                                ]),
                                                html.Label([
                                                    'Metadata value: ',
                                                    dcc.Dropdown(
                                                        id='metadata-value',
                                                        options={
                                                        },
                                                        placeholder='Select value...',
                                                    ),
                                                ]),
                                                html.Br(),
                                            ]),

                                            html.Div(children=[
                                                html.Button('Next Step', className='operations-button',
                                                            id={'type': 'next-button', 'index': 1}),
                                                html.A(html.Button('Reset Application', className='operations-button'),
                                                       href='javascript:window.location.href=/^[^?]+/.exec(window.location.href)[0]'),
                                            ])

                                        ])
                                    ]
                                ),
                                dcc.Tab(
                                    label='Step 2 \n Alignment',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    # id='alignment-block',
                                    children=[
                                        html.Div(id='alignment-block', children=[
                                            html.Div([
                                                html.H3('Dataset Alignment', className='block-title'),
                                                html.Label([
                                                    'Alignment Algorithm:',
                                                    html.I(id='alignment-tooltip',
                                                           className='fa-regular fa-circle-question'),
                                                    dcc.Dropdown(id='alignment-method', options=[
                                                        {'label': ['Linear Manifold Alignment'], 'value': 'lma'},
                                                        {'label': ['Nonlinear Manifold Alignment'], 'value': 'nlma'},
                                                        {'label': ['Canonical Correlation Analysis'], 'value': 'cca'},
                                                        {'label': ['Manifold Alignment with Maximum Mean Discrepancy'], 'value': 'mmdma', 'disabled': False},
                                                        {'label': ['UnionCom'], 'value': 'unioncom', 'disabled': False},
                                                    ]),
                                                    dbc.Tooltip([
                                                        'Alignment Algorithms:',
                                                        html.Ul([
                                                            html.Li([
                                                                html.A('Linear manifold alignment', href='https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6329-2'),
                                                                ' is best for applications involving modalities which are closely related.  The linear nature of the method will prevent overfitting, but can miss more complex relationships.']),
                                                            html.Li([
                                                                html.A('Nonlinear manifold alignment', href='https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6329-2'),
                                                                ' generates an embedding function mapping each modality onto the same manifold, allowing for interpretation of complex relationships.  NLMA will closely match associated cells, but might sacrifice cluster accuracy to do so.'
                                                            ]),
                                                            html.Li([
                                                                html.A('Canonical correlation analysis', href='https://doi.org/10.2307/2333955'),
                                                                ' is a robust method which specializes in characterizing the variance of the data overall.  This is best for modalities with few, closely related features.'
                                                            ]),
                                                            html.Li([
                                                                html.A('Manifold Alignment with Maximum Mean Discrepancy', href='https://pubmed.ncbi.nlm.nih.gov/34632462/'),
                                                                ' is an iterative machine learning method which aligns modalities using a series of counterbalanced losses.  MMD-MA is particularly good at identifying cell type clusters, but takes comparatively long to run.'
                                                                ]),
                                                            html.Li([
                                                                html.A('UnionCom', href='https://doi.org/10.1093/bioinformatics/btaa443'),
                                                                ' infers cross-modal correspondence before projecting each modality to similar latent spaces.  UnionCom is particularly good with aligning unmatched modalities, but takes a comparatively long time to run.']
                                                            )
                                                        ])
                                                    ], target='alignment-tooltip', trigger='legacy'),
                                                ]),
                                                html.Label([
                                                    'Latent space dimension: ',
                                                    # dcc.Input(id='ndims', value='5', style={'width': '20px'},
                                                    #           type='number', min=3, max=20, step=1),
                                                    dcc.Slider(id='ndims', value=5, min=3, max=10, step=1)
                                                ]),
                                                html.Br(),
                                                html.Label([
                                                    'Number of nearest neighbors: ',
                                                    # dcc.Input(id='neighbors', value='2', style={'width': '20px'},
                                                    #           type='number', min=1, max=10, step=1),
                                                    dcc.Slider(id='neighbors', value=2, min=1, max=10, step=1)
                                                ]),
                                                html.Label([
                                                    'Iterations (MMD-MA): ',
                                                    # dcc.Input(id='neighbors', value='2', style={'width': '20px'},
                                                    #           type='number', min=1, max=10, step=1),
                                                    dcc.Dropdown(id='mmdma_iterations', value=1000,
                                                                 options=[200, 500, 1000, 2000, 5000, 10000])
                                                ]),
                                                html.Button('Align Datasets', id='btn-align',
                                                            className='operations-button'),
                                                html.Button('Download Aligned Data', id='btn-align-download',
                                                            className='operations-button'),
                                                dcc.Download(id='download-aligned')
                                            ]),
                                            html.Div([
                                                html.Div(className='obh-container', children=[
                                                    html.Button('Prev Step', className='operations-button-half',
                                                                id={'type': 'prev-button', 'index': 2}),
                                                    html.Button('Next Step', className='operations-button-half',
                                                                id={'type': 'next-button', 'index': 2})
                                                    ]),
                                                html.A(html.Button('Reset Application', className='operations-button'),
                                                       href='javascript:window.location.href=/^[^?]+/.exec(window.location.href)[0]'),
                                            ])
                                        ]),



                                    ]
                                ),

                                dcc.Tab(
                                    label='Step 3 \n Clustering',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    # id='clustering-block',
                                    children=[
                                        html.Div(id='clustering-block', children=[
                                            html.Div([
                                                html.H3('Cross-modal clustering', className='block-title'),
                                                html.Label(
                                                    children=[
                                                        'Clustering Algorithm: ',
                                                        html.I(id='clustering-tooltip',
                                                               className='fa-regular fa-circle-question'),
                                                        dcc.RadioItems(
                                                            id='clustering-method',
                                                            options={
                                                                'gmm': 'Gaussian Mixture Model',
                                                                'kmeans': 'k-means',
                                                                'hierarchical': 'Hierarchical'
                                                            },
                                                            value='gmm'
                                                        ),
                                                    ]
                                                ),
                                                dbc.Tooltip([
                                                    'Clustering Algorithms: (see ',
                                                    html.A('https://scikit-learn.org/stable/modules/clustering.html', href='https://scikit-learn.org/stable/modules/clustering.html'),
                                                    ')',
                                                    html.Ul([
                                                        html.Li('Gaussian Mixture Model: A probabilistic clustering algorithm similar to K-means which is capable of classifying elliptical groupings.'),
                                                        html.Li('K-means: A very explainable, lightweight algorithm that works best with even-sized circular clusters.'),
                                                        html.Li('Hierarchical: Point-wise agglomerative clustering technique which works best for separable clusters of non-elliptical shapes.')
                                                    ])
                                                ], target='clustering-tooltip', trigger='legacy'),
                                                html.Br(),
                                                html.Label([
                                                    'Number of clusters: ',
                                                    # dcc.Input(id='num_clusters', value='5', style={'width': '20px'},
                                                    #           type='number', min=1, max=10, step=1),
                                                    dcc.Slider(id='num_clusters', value=5, min=1, max=10, step=1)
                                                ]),
                                                html.Button('Identify Cross-Modal Cell Clusters', id='btn-cluster',
                                                            className='operations-button'),
                                                html.Button('Download Clusters', id='btn-cluster-download',
                                                            className='operations-button'),
                                                dcc.Download(id='download-cluster'),
                                            ]),
                                            html.Div([
                                                html.Div(className='obh-container', children=[
                                                    html.Button('Prev Step', className='operations-button-half',
                                                                id={'type': 'prev-button', 'index': 3}),
                                                    html.Button('Next Step', className='operations-button-half',
                                                                id={'type': 'next-button', 'index': 3}),
                                                ]),
                                                html.A(html.Button('Reset Application', className='operations-button'),
                                                       href='javascript:window.location.href=/^[^?]+/.exec(window.location.href)[0]'),
                                            ])
                                        ])

                                    ]
                                ),

                                dcc.Tab(
                                    label='Step 4 \n Analysis',
                                    className='custom-tab',
                                    selected_className='custom-tab-selected',
                                    # id='controls-block',
                                    children=[
                                        html.Div(id='controls-block', children=[
                                            html.Div([
                                                html.H4('Plot type'),
                                                dcc.Dropdown(
                                                    id='plot-type',
                                                    options={
                                                        'alignment-combo': 'Alignment',
                                                        'separate2': 'Aligned Cells (2D)',
                                                        'separate3': 'Aligned Cells (3D)',
                                                        'bibiplot': 'Top Feature Correlation with Latent Space (Bibiplot)',
                                                        'heatmap2': 'Features of Cross-modal Clusters (Heatmap)'
                                                    },
                                                    value='alignment'
                                                ),

                                                html.H4('Color by...'),
                                                dcc.RadioItems(
                                                    id='color-type',
                                                    options=color_types,
                                                    value='cluster'
                                                ),
                                                dcc.Dropdown(
                                                    id='metadata-type',
                                                    options={
                                                        'ttype': 'Transcriptomic Type'
                                                    },
                                                    placeholder='Select metadata...',
                                                ),
                                                html.H4('Component Selection',
                                                        title='Select components of the aligned datasets to be plotted .'),
                                                html.Label([' X: ',
                                                            dcc.Input(id='component_x',
                                                                      type='number',
                                                                      min=1,
                                                                      step=1,
                                                                      value=1,
                                                                      className='component_input')]),
                                                html.Label([' Y: ',
                                                            dcc.Input(id='component_y',
                                                                      type='number',
                                                                      min=1,
                                                                      step=1,
                                                                      value=2,
                                                                      className='component_input')]),
                                                html.Label([' Z: ',
                                                            dcc.Input(id='component_z',
                                                                      type='number',
                                                                      min=1,
                                                                      step=1,
                                                                      value=3,
                                                                      className='component_input')]),
                                                html.H3('Features of Cross-modal Clusters'),
                                                html.H4([
                                                    'Number of Top Features per cluster ',
                                                    dcc.Input(id='num_enriched',
                                                              type='number',
                                                              min=1,
                                                              step=1,
                                                              value=10,
                                                              className='component_input'),
                                                ]),
                                                html.H3('Top Feature Correlation with Latent Space'),
                                                html.H4([
                                                    'Number of Top Correlated Features ',
                                                    dcc.Input(id='num_correlated',
                                                              type='number',
                                                              min=1,
                                                              step=1,
                                                              value=10,
                                                              className='component_input'),
                                                ]),

                                                dcc.Checklist(id='hires-plots',
                                                              options=[{'label': 'High-resolution plots', 'value': 'hires'}]),
                                                html.Button('Refresh Plot',
                                                            id='btn-refresh-plot',
                                                            className='operations-button'),
                                                html.Button('Download Top Features',
                                                            id='btn-enriched-download',
                                                            className='operations-button'),
                                                dcc.Download(id='download-enriched'),
                                            ]),
                                            html.Div([
                                                html.Button('Prev Step', className='operations-button',
                                                            id={'type': 'prev-button', 'index': 4}),
                                                html.A(html.Button('Reset Application', className='operations-button'),
                                                       href='javascript:window.location.href=/^[^?]+/.exec(window.location.href)[0]'),
                                            ])
                                        ])

                                    ]
                                )
                            ]
                        ),
                    ]

                ),



                html.Div(
                    id='right-panel',
                    children=[
                        dbc.Alert(id='user-data-alert',
                                  children=[],
                                  color='primary',
                                  is_open=False,
                                  dismissable=True),
                        dbc.Alert(id='user-data-alert-x',
                                  children=[],
                                  color='danger',
                                  is_open=False,
                                  dismissable=True),
                        dbc.Alert(id='user-data-alert-bg',
                                  children=[],
                                  color='info',
                                  is_open=False,
                                  dismissable=True),
                        dbc.Alert(id='user-data-alert-dl',
                                  children=[],
                                  color='info',
                                  is_open=False,
                                  dismissable=True),
                        html.Div(
                            id='upload-page',
                            children=[
                                html.Div(
                                    id='graph_block-x',
                                    children=[
                                        dcc.Graph(
                                            id='graph-x',
                                            mathjax=True,
                                            figure=go.Figure(data={}, layout=blank_layout),
                                            style={'height': '600px', 'width': '600px'}
                                        ),
                                    ]
                                ),
                                html.P(id='graph_legend-x', children='')
                            ]
                        ),
                        html.Div(
                            id='graph-page',
                            children=[
                                html.Div(
                                    id='graph_block',
                                    children=[
                                        dcc.Graph(
                                            id='graph-combined',
                                            mathjax=True,
                                            figure=go.Figure(data={}, layout=blank_layout),
                                            style={'height': '600px', 'width': '600px'}
                                        ),
                                    ]
                                ),
                                html.P(id='graph_legend', children='')
                            ]
                        ),
                        html.Div(
                            id='about-page',
                            children=[

                                # # Background callback simple example
                                # html.Div([
                                #     html.Div([html.P(id="bg_paragraph_id", children=["Button not clicked"])]),
                                #     html.Button(id="bg_button_id", children="Run Job!"),
                                # ]),


                                html.Img(src='',
                                         alt='Manifold alignment and clustering flow chart',
                                         className='tab-image', id='flowchart-img'),
                                # html.P(children=[
                                #     'This application demonstrates use of nonlinear manifold learning to '
                                #     'align single cell multi-modal data sets in a low-dimension latent space, '
                                #     'enabling identification of cross-modal cell clusters.'
                                # ]),
                                html.H3('Usage'),
                                html.Ol(id='list-steps',
                                    children=[
                                    html.Li([
                                        html.H4('Upload Data'),
                                        html.P(['On the Data tab, select a predefined data set (mouse motor or '
                                               'visual cortex neurons) or upload your own data files. '
                                               'User uploaded data must be in .csv files where the first row '
                                               'contains column names and the first column contains cell '
                                               'identifiers. ',
                                               'Files must have consistent cell (row) order.']),
                                        html.Ol(children=[
                                            html.Li(
                                                'Modality 1 single-cell data file (e.g. gene expression, neuron morphology, or electrophysiological feature)'),
                                            html.Li('Modality 2 single-cell data file'),
                                            html.Li('Single-cell metadata/phenotype data, expected to be categorical')]),
                                        html.P(['The first 1000 rows of uploaded data files are used.  Larger datasets will be truncated, so we recommend reducing the data size by ',
                                                'computation of ', html.A('metacells', href='https://doi.org/10.1186/s13059-019-1812-2'),
                                                ' or ', html.A('SEACells', href='https://doi.org/10.1038/s41587-023-01716-9'),
                                                ' before upload.']),
                                        html.P([
                                            'Sample datasets may be downloaded here: ',
                                            html.Button('Mouse motor cortex (Gene expression, electrophysiology)',
                                                        id='sample-data-download-button1'),
                                            ' ',
                                            html.Button('Mouse motor cortex (Gene expression, morphology)',
                                                        id='sample-data-download-button2'),
                                            dcc.Download(id='download-sample-data')]

                                        ),

                                        html.H5('Preprocessing'),
                                        html.P('Optionally, select a preprocessing operation to be performed '
                                               'on each modality prior to alignment.  The currently supported '
                                               'operations are log transformation and standardization.'),

                                        html.H5('Explore data'),
                                        html.P('The selected dataset can be explored by entering one or two cell features in the "X, Y" box along with '
                                               'a metadata type.  If desired, the set of cells can be filtered by selecting a metadata value. '
                                               'The feature name can be typed or selected from a drop-down box and copied using the "+" button.'),
                                        html.P('If a single feature is entered, the distribution of values will be shown as a set of box plots, grouped by metadata value. '
                                               'If two features are entered, then they will be plotted against each other in a scatter plot, colored by metadata value.')
                                    ]),
                                    html.Li([
                                        html.H4('Align Data'),
                                        html.P('The alignment tab provides controls to select an alignment method and '
                                               'the dimension '
                                               'of the latent space to which the cellular data should be projected. '
                                               'The Linear Manifold Alignment and Nonlinear Manifold Alignment methods will '
                                               'also use the number of nearest neighbors to be used in constructing '
                                               'similarity matrices.'),
                                        html.P('Available alignment methods include:'),
                                        html.Ul([
                                            html.Li([
                                                html.A('Linear manifold alignment',
                                                       href='https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6329-2'),
                                                ' is best for applications involving modalities which are closely related.  The linear nature of the method will prevent overfitting, but can miss more complex relationships.']),
                                            html.Li([
                                                html.A('Nonlinear manifold alignment',
                                                       href='https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6329-2'),
                                                ' generates an embedding function mapping each modality onto the same manifold, allowing for interpretation of complex relationships.  NLMA will closely match associated cells, but might sacrifice cluster accuracy to do so.'
                                            ]),
                                            html.Li([
                                                html.A('Canonical correlation analysis',
                                                       href='https://doi.org/10.2307/2333955'),
                                                ' is a robust method which specializes in characterizing the variance of the data overall.  This is best for modalities with few, closely related features.'
                                            ]),
                                            html.Li([
                                                html.A('Manifold Alignment with Maximum Mean Discrepancy',
                                                       href='https://pubmed.ncbi.nlm.nih.gov/34632462/'),
                                                ' is an iterative machine learning method which aligns modalities using a series of counterbalanced losses.  MMD-MA is particularly good at identifying cell type clusters, but takes comparatively long to run.'
                                            ]),
                                            html.Li([
                                                html.A('UnionCom',
                                                       href='https://doi.org/10.1093/bioinformatics/btaa443'),
                                                ' infers cross-modal correspondence before projecting each modality to similar latent spaces.  UnionCom is particularly good with aligning unmatched modalities, but takes a comparatively long time to run.']
                                            )
                                        ]),

                                        html.P('Click the "Align Datasets" button to perform the alignment. '
                                               'If the alignment parameters are changed, the "Align Datasets" '
                                               'button will need to be used again to recompute the alignment.'),
                                        html.P('Once the alignment has been performed, the aligned data can be '
                                               'downloaded by clicking on the "Download Aligned Data" button.')]),
                                    html.Li([
                                        html.H4('Clustering'),
                                        html.P('The clustering tab includes controls to select a clustering algorithm '
                                               'as well as the number of cross-modal cell clusters which should be '
                                               'identified.  Click the '
                                               '"Download Clusters" button to download files that label '
                                               'cells with the identified cross-modal clusters.'),
                                        html.P('Available clustering methods include:'),
                                        html.Ul([
                                            html.Li('Gaussian Mixture Model: '
                                                    'A probabilistic clustering algorithm similar to K-means which is capable of classifying elliptical groupings.'),
                                            html.Li('K-means: ' 
                                                    'A very explainable, lightweight algorithm that works best with even-sized circular clusters.'),
                                            html.Li('Hierarchical: '
                                                    'Point-wise agglomerative clustering technique which works best for separable clusters of non-elliptical shapes.'),
                                        ])
                                    ]),
                                    html.Li([
                                        html.H4('Analysis'),
                                        html.P('The analysis tab allows selection of visualizations '
                                               'and control of relevant parameters. '
                                               'Supported plot types include:'),
                                        html.Ul([
                                            html.Li([
                                                html.H5('Alignment'),
                                                html.Span('Display a 3-D projection of the aligned datasets '
                                                          'overlaid in the latent space along with '
                                                          'box plots characterizing alignment error '
                                                          'using pairwise cell distance and Fraction of Samples '
                                                          'Closer Than True Match (FOSCTTM).')
                                            ]),
                                            html.Li([
                                                html.H5('Aligned Cells (2D)'),
                                                html.Span('Display a 2-D projection of the aligned datasets '
                                                          'in two side-by-side plots.')
                                            ]),
                                            html.Li([
                                                html.H5('Aligned Cells (3D)'),
                                                html.Span('Display a 3-D projection of the aligned datasets '
                                                          'in two side-by-side plots.')
                                            ]),
                                            html.Li([
                                                html.H5('Top Feature Correlation with Latent Space (Bibiplot)'),
                                                html.Span('Display biplots in two dimensions of the latent space. '
                                                          'Aligned datasets are projected along with lines '
                                                          'describing the correlation of features with their '
                                                          'latent space representation.')
                                            ]),
                                            html.Li([
                                                html.H5('Features of Cross-modal Clusters (Heatmap)'),
                                                html.Span('Computes and plots the top features by cross-modal cluster.')
                                            ]),

                                        ]),
                                        html.P('Controls are provided to control cell coloring (by cross-modal '
                                               'cluster or cell metadata) and to select the '
                                               'latent space components which should be used in creating the '
                                               'above visualizations.  The number of top features to be identified for '
                                               'each cross-modal cluster may also be specified.')
                                    ]),
                                ])
                            ]
                        )

                    ]
                ),
                html.Div(id='progress_container', children=[
                    dcc.Loading(id='loading-1',
                                children=[html.Div(id='loading-alignment'),
                                          html.Div(id='loading-alignment-bg'),
                                          html.Div(id='loading-plot'),
                                          html.Div(id={'type': 'loading-upload', 'index': UploadFileType.DATA_1}),
                                          html.Div(id={'type': 'loading-upload', 'index': UploadFileType.DATA_2}),
                                          html.Div(id={'type': 'loading-upload', 'index': UploadFileType.METADATA}),
                                          ]),
                    # dcc.Loading(id='loading-2',
                    #             children=html.Div(id='loading-plot', className='loading_trigger')),
                    # id={'type': 'dynamic-output', 'index': UploadFileType.DATA_2}
                ])

            ]
        ),

        html.Div(
            id='id_footer',
            children=[
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
                
                 <div class="uw-logo">
                 <a href="http://www.wisc.edu" aria-label="Link to main university website">
                 <svg aria-hidden="true" focusable="false">
                     <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#uw-symbol-crest-footer"></use>
                 </svg>
                 </a>
                 </div>
                
    <!-- SVG icons -->
    <svg aria-hidden="true" style="display: inline; height: 0; position: absolute;" xmlns:xlink="http://www.w3.org/1999/xlink">
      <title>Site icons</title>
      <defs>
        <symbol id="uw-symbol-more" viewBox="0,0,1792,1792">
          <title id="title">More</title>
          <path d="M979 960q0 13-10 23l-466 466q-10 10-23 10t-23-10l-50-50q-10-10-10-23t10-23l393-393-393-393q-10-10-10-23t10-23l50-50q10-10 23-10t23 10l466 466q10 10 10 23zm384 0q0 13-10 23l-466 466q-10 10-23 10t-23-10l-50-50q-10-10-10-23t10-23l393-393-393-393q-10-10-10-23t10-23l50-50q10-10 23-10t23 10l466 466q10 10 10 23z"></path>
        </symbol>
        <symbol id="uw-symbol-menu" viewBox="0 0 1024 1024">
          <title id="svg-menu">open menu</title>
          <path class="path1" d="M128 256h768v86h-768v-86zM128 554v-84h768v84h-768zM128 768v-86h768v86h-768z"></path>
        </symbol>
        <symbol id="uw-symbol-close" viewBox="0 0 805 1024">
          <title>close</title>
          <path class="path1" d="M741.714 755.429q0 22.857-16 38.857l-77.714 77.714q-16 16-38.857 16t-38.857-16l-168-168-168 168q-16 16-38.857 16t-38.857-16l-77.714-77.714q-16-16-16-38.857t16-38.857l168-168-168-168q-16-16-16-38.857t16-38.857l77.714-77.714q16-16 38.857-16t38.857 16l168 168 168-168q16-16 38.857-16t38.857 16l77.714 77.714q16 16 16 38.857t-16 38.857l-168 168 168 168q16 16 16 38.857z"></path>
        </symbol>
        <symbol id="uw-symbol-search" viewBox="0 0 951 1024">
          <title>search</title>
          <path class="path1" d="M658.286 475.429q0-105.714-75.143-180.857t-180.857-75.143-180.857 75.143-75.143 180.857 75.143 180.857 180.857 75.143 180.857-75.143 75.143-180.857zM950.857 950.857q0 29.714-21.714 51.429t-51.429 21.714q-30.857 0-51.429-21.714l-196-195.429q-102.286 70.857-228 70.857-81.714 0-156.286-31.714t-128.571-85.714-85.714-128.571-31.714-156.286 31.714-156.286 85.714-128.571 128.571-85.714 156.286-31.714 156.286 31.714 128.571 85.714 85.714 128.571 31.714 156.286q0 125.714-70.857 228l196 196q21.143 21.143 21.143 51.429z"></path>
        </symbol>
        <symbol id="uw-symbol-search-gray" viewBox="0 0 951 1024">
          <title>search</title>
          <path class="path1" d="M658.286 475.429q0-105.714-75.143-180.857t-180.857-75.143-180.857 75.143-75.143 180.857 75.143 180.857 180.857 75.143 180.857-75.143 75.143-180.857zM950.857 950.857q0 29.714-21.714 51.429t-51.429 21.714q-30.857 0-51.429-21.714l-196-195.429q-102.286 70.857-228 70.857-81.714 0-156.286-31.714t-128.571-85.714-85.714-128.571-31.714-156.286 31.714-156.286 85.714-128.571 128.571-85.714 156.286-31.714 156.286 31.714 128.571 85.714 85.714 128.571 31.714 156.286q0 125.714-70.857 228l196 196q21.143 21.143 21.143 51.429z"></path>
        </symbol>
        <symbol id="uw-symbol-search-in-blank" viewBox="0 0 951 1024">
          <title>search</title>
          <path class="path1" d="M658.286 475.429q0-105.714-75.143-180.857t-180.857-75.143-180.857 75.143-75.143 180.857 75.143 180.857 180.857 75.143 180.857-75.143 75.143-180.857zM950.857 950.857q0 29.714-21.714 51.429t-51.429 21.714q-30.857 0-51.429-21.714l-196-195.429q-102.286 70.857-228 70.857-81.714 0-156.286-31.714t-128.571-85.714-85.714-128.571-31.714-156.286 31.714-156.286 85.714-128.571 128.571-85.714 156.286-31.714 156.286 31.714 128.571 85.714 85.714 128.571 31.714 156.286q0 125.714-70.857 228l196 196q21.143 21.143 21.143 51.429z"></path>
        </symbol>
        <symbol id="uw-symbol-envelope" viewBox="0 0 1024 1024">
          <title>envelope</title>
          <path class="path1" d="M1024 405.714v453.714q0 37.714-26.857 64.571t-64.571 26.857h-841.143q-37.714 0-64.571-26.857t-26.857-64.571v-453.714q25.143 28 57.714 49.714 206.857 140.571 284 197.143 32.571 24 52.857 37.429t54 27.429 62.857 14h1.143q29.143 0 62.857-14t54-27.429 52.857-37.429q97.143-70.286 284.571-197.143 32.571-22.286 57.143-49.714zM1024 237.714q0 45.143-28 86.286t-69.714 70.286q-214.857 149.143-267.429 185.714-5.714 4-24.286 17.429t-30.857 21.714-29.714 18.571-32.857 15.429-28.571 5.143h-1.143q-13.143 0-28.571-5.143t-32.857-15.429-29.714-18.571-30.857-21.714-24.286-17.429
          q-52-36.571-149.714-104.286t-117.143-81.429q-35.429-24-66.857-66t-31.429-78q0-44.571 23.714-74.286t67.714-29.714h841.143q37.143 0 64.286 26.857t27.143 64.571z"></path>
        </symbol>
        <symbol id="uw-symbol-pinterest" class="uw-social-symbols" viewBox="0 0 731 1024">
          <title>pinterest-p</title>
          <path class="path1" d="M0 341.143q0-61.714 21.429-116.286t59.143-95.143 86.857-70.286 105.714-44.571 115.429-14.857q90.286 0 168 38t126.286 110.571 48.571 164q0 54.857-10.857 107.429t-34.286 101.143-57.143 85.429-82.857 58.857-108 22q-38.857 0-77.143-18.286t-54.857-50.286q-5.714 22.286-16 64.286t-13.429 54.286-11.714 40.571-14.857 40.571-18.286 35.714-26.286 44.286-35.429 49.429l-8 2.857-5.143-5.714q-8.571-89.714-8.571-107.429 0-52.571 12.286-118t38-164.286 29.714-116q-18.286-37.143-18.286-96.571 0-47.429 29.714-89.143t75.429-41.714q34.857 0 54.286 23.143t19.429 58.571
          q0 37.714-25.143 109.143t-25.143 106.857q0 36 25.714 59.714t62.286 23.714q31.429 0 58.286-14.286t44.857-38.857 32-54.286 21.714-63.143 11.429-63.429 3.714-56.857q0-98.857-62.571-154t-163.143-55.143q-114.286 0-190.857 74t-76.571 187.714q0 25.143 7.143 48.571t15.429 37.143 15.429 26 7.143 17.429q0 16-8.571 41.714t-21.143 25.714q-1.143 0-9.714-1.714-29.143-8.571-51.714-32t-34.857-54-18.571-61.714-6.286-60.857z"></path>
        </symbol>
        <symbol id="uw-symbol-twitter" class="uw-social-symbols" viewBox="0 0 951 1024">
          <title>twitter</title>
          <path class="path1" d="M925.714 233.143q-38.286 56-92.571 95.429 0.571 8 0.571 24 0 74.286-21.714 148.286t-66 142-105.429 120.286-147.429 83.429-184.571 31.143q-154.857 0-283.429-82.857 20 2.286 44.571 2.286 128.571 0 229.143-78.857-60-1.143-107.429-36.857t-65.143-91.143q18.857 2.857 34.857 2.857 24.571 0 48.571-6.286-64-13.143-106-63.714t-42-117.429v-2.286q38.857 21.714 83.429 23.429-37.714-25.143-60-65.714t-22.286-88q0-50.286 25.143-93.143 69.143 85.143 168.286 136.286t212.286 56.857q-4.571-21.714-4.571-42.286 0-76.571 54-130.571t130.571-54
          q80 0 134.857 58.286 62.286-12 117.143-44.571-21.143 65.714-81.143 101.714 53.143-5.714 106.286-28.571z"></path>
        </symbol>
        <symbol id="uw-symbol-youtube" class="uw-social-symbols" viewBox="0 0 878 1024">
          <title>youtube</title>
          <path class="path1" d="M554.857 710.857v120.571q0 38.286-22.286 38.286-13.143 0-25.714-12.571v-172q12.571-12.571 25.714-12.571 22.286 0 22.286 38.286zM748 711.429v26.286h-51.429v-26.286q0-38.857 25.714-38.857t25.714 38.857zM196 586.857h61.143v-53.714h-178.286v53.714h60v325.143h57.143v-325.143zM360.571 912h50.857v-282.286h-50.857v216q-17.143 24-32.571 24-10.286 0-12-12-0.571-1.714-0.571-20v-208h-50.857v223.429q0 28 4.571 41.714 6.857 21.143 33.143 21.143 27.429 0 58.286-34.857v30.857zM605.714 827.429v-112.571q0-41.714-5.143-56.571-9.714-32-40.571-32-28.571 0-53.143 30.857
          v-124h-50.857v378.857h50.857v-27.429q25.714 31.429 53.143 31.429 30.857 0 40.571-31.429 5.143-15.429 5.143-57.143zM798.857 821.714v-7.429h-52q0 29.143-1.143 34.857-4 20.571-22.857 20.571-26.286 0-26.286-39.429v-49.714h102.286v-58.857q0-45.143-15.429-66.286-22.286-29.143-60.571-29.143-38.857 0-61.143 29.143-16 21.143-16 66.286v98.857q0 45.143 16.571 66.286 22.286 29.143 61.714 29.143 41.143 0 61.714-30.286 10.286-15.429 12-30.857 1.143-5.143 1.143-33.143zM451.429 300
          v-120q0-39.429-24.571-39.429t-24.571 39.429v120q0 40 24.571 40t24.571-40zM862.286 729.143q0 133.714-14.857 200-8 33.714-33.143 56.571t-58.286 26.286q-105.143 12-317.143 12t-317.143-12q-33.143-3.429-58.571-26.286t-32.857-56.571q-14.857-64-14.857-200 0-133.714 14.857-200 8-33.714 33.143-56.571t58.857-26.857q104.571-11.429 316.571-11.429t317.143 11.429q33.143 4 58.571 26.857t32.857 56.571q14.857 64 14.857 200zM292 0h58.286l-69.143 228v154.857h-57.143v-154.857q-8-42.286-34.857-121.143-21.143-58.857-37.143-106.857h60.571l40.571 150.286zM503.429 190.286
          v100q0 46.286-16 67.429-21.143 29.143-60.571 29.143-38.286 0-60-29.143-16-21.714-16-67.429v-100q0-45.714 16-66.857 21.714-29.143 60-29.143 39.429 0 60.571 29.143 16 21.143 16 66.857zM694.857 97.714v285.143h-52v-31.429q-30.286 35.429-58.857 35.429-26.286 0-33.714-21.143-4.571-13.714-4.571-42.857v-225.143h52v209.714q0 18.857 0.571 20 1.714 12.571 12 12.571 15.429 0 32.571-24.571v-217.714h52z"></path>
        </symbol>
        <symbol id="uw-symbol-facebook" class="uw-social-symbols" viewBox="0 0 602 1024">
          <title>facebook</title>
          <path class="path1" d="M548 6.857v150.857h-89.714q-49.143 0-66.286 20.571t-17.143 61.714v108h167.429l-22.286 169.143h-145.143v433.714h-174.857v-433.714h-145.714v-169.143h145.714v-124.571q0-106.286 59.429-164.857t158.286-58.571q84 0 130.286 6.857z"></path>
        </symbol>
        <symbol id="uw-symbol-instagram" class="uw-social-symbols" viewBox="0 0 878 1024">
          <title>instagram</title>
          <path class="path1" d="M778.286 814.857v-370.286h-77.143q11.429 36 11.429 74.857 0 72-36.571 132.857t-99.429 96.286-137.143 35.429q-112.571 0-192.571-77.429t-80-187.143q0-38.857 11.429-74.857h-80.571v370.286q0 14.857 10 24.857t24.857 10h610.857q14.286 0 24.571-10t10.286-24.857zM616 510.286q0-70.857-51.714-120.857t-124.857-50q-72.571 0-124.286 50t-51.714 120.857 51.714 120.857 124.286 50q73.143 0 124.857-50t51.714-120.857zM778.286 304.571v-94.286q0-16-11.429-27.714t-28-11.714h-99.429q-16.571 0-28 11.714t-11.429 27.714v94.286q0 16.571 11.429 28
          t28 11.429h99.429q16.571 0 28-11.429t11.429-28zM877.714 185.714v652.571q0 46.286-33.143 79.429t-79.429 33.143h-652.571q-46.286 0-79.429-33.143t-33.143-79.429v-652.571q0-46.286 33.143-79.429t79.429-33.143h652.571q46.286 0 79.429 33.143t33.143 79.429z"></path>
        </symbol>
        <symbol id="uw-symbol-caret-up" viewBox="0 0 1792 1792">
          <title>Collapse</title>
          <path d="M1395 1184q0 13-10 23l-50 50q-10 10-23 10t-23-10l-393-393-393 393q-10 10-23 10t-23-10l-50-50q-10-10-10-23t10-23l466-466q10-10 23-10t23 10l466 466q10 10 10 23z"></path>
        </symbol>
        <symbol id="uw-symbol-caret-down" viewBox="0 0 1792 1792">
          <title>Expand</title>
          <path d="M1395 736q0 13-10 23l-466 466q-10 10-23 10t-23-10l-466-466q-10-10-10-23t10-23l50-50q10-10 23-10t23 10l393 393 393-393q10-10 23-10t23 10l50 50q10 10 10 23z"></path>
        </symbol>
        <symbol id="uw-symbol-chevron-right" viewBox="0 0 695 1024">
          <title id="svg-next-slide">next slide</title>
          <path class="path1" d="M632.571 501.143l-424 424q-10.857 10.857-25.714 10.857t-25.714-10.857l-94.857-94.857q-10.857-10.857-10.857-25.714t10.857-25.714l303.429-303.429-303.429-303.429q-10.857-10.857-10.857-25.714t10.857-25.714l94.857-94.857q10.857-10.857 25.714-10.857t25.714 10.857l424 424q10.857 10.857 10.857 25.714t-10.857 25.714z"></path>
        </symbol>
        <symbol id="uw-symbol-chevron-left" viewBox="0 0 768 1024">
          <title id="svg-previous">previous slide</title>
          <path class="path1" d="M669.143 172l-303.429 303.429 303.429 303.429q10.857 10.857 10.857 25.714t-10.857 25.714l-94.857 94.857q-10.857 10.857-25.714 10.857t-25.714-10.857l-424-424q-10.857-10.857-10.857-25.714t10.857-25.714l424-424q10.857-10.857 25.714-10.857t25.714 10.857l94.857 94.857q10.857 10.857 10.857 25.714t-10.857 25.714z"></path>
        </symbol>
        <symbol id="uw-symbol-crest-footer" viewBox="0 0 200 132.78">
          <title id="svg-crest-footer">Footer Crest</title>
          <path d="M31.65,93.63c3,1.09,1.75,4.35,1,6.65L28.31,115l-5.4-17.44a8.47,8.47,0,0,1-.58-4.07H16.85v0.12c2.14,1.13,1.36,3.75.62,6.2l-4.55,15.31L7.14,96.41a7.23,7.23,0,0,1-.47-2.9H1v0.12c1.94,1.37,2.53,4,3.23,6.2l4.58,14.86c1.28,4.15,1.63,3.87,5.16,6.53L20,100.88l4.27,13.86c1.29,4.15,1.56,3.95,5.13,6.49l8.19-27.71h-6v0.12Z" 
transform="translate(-1 -0.61)"/><path d="M38.95,93.63c2,0.77,1.71,3.71,1.71,5.56v15.18c0,1.81.23,4.8-1.71,5.52V120h7.38v-0.12c-1.94-.77-1.71-3.71-1.71-5.52V99.19c0-1.81-.23-4.79,1.71-5.56V93.51H38.95v0.12Z" 
transform="translate(-1 -0.61)"/><path d="M53.12,99.43c0-2.78,2.8-3.67,5-3.67a8.68,8.68,0,0,1,6.1,2.54V93.75a18.9,18.9,0,0,0-5.79-.89c-4.74,0-9.75,1.94-9.75,7,0,8.54,13.36,7.41,13.36,13.7,0,2.82-3.34,4.19-5.55,4.19A13.17,13.17,0,0,1,48,114.41l1.13,5a19.44,19.44,0,0,0,7,1.21c6.06,0,10.41-4,10.41-8.34C66.49,104.87,53.12,104.95,53.12,99.43Z" 
transform="translate(-1 -0.61)"/><path d="M82.23,117.32c-6,0-9.87-5.28-9.87-11.2s3.73-9.91,9.09-9.91a13.15,13.15,0,0,1,7.19,2.3V93.87a30,30,0,0,0-7.07-1c-8,0-13.64,5.52-13.64,13.86,0,8.62,5.67,13.94,14.84,13.94a13.31,13.31,0,0,0,5.86-1.21l2-4.67H90.5A16.7,16.7,0,0,1,82.23,117.32Z" 
transform="translate(-1 -0.61)"/><path d="M104.63,92.83a14,14,0,0,0-14.57,13.85,13.83,13.83,0,0,0,14.18,14,14.09,14.09,0,0,0,14.29-14.18A13.63,13.63,0,0,0,104.63,92.83Zm-0.19,24.93c-6.1,0-9.95-5.8-9.95-11.44,0-6.08,3.85-10.59,9.87-10.59s9.75,5.68,9.75,11.56S110.34,117.75,104.44,117.75Z" 
transform="translate(-1 -0.61)"/><path d="M138.49,93.63c2.06,1.21,2,2.82,2,5.08V114L126.29,95.36l-0.55-.81a2.48,2.48,0,0,1-.58-1v0H119.5v0.12c2,1.21,2,2.82,2,5.08v16.07c0,2.25,0,3.86-2.06,5.11V120h6.88v-0.12c-2.06-1.25-2-2.86-2-5.11V99.19l13.32,17.52c1.71,2.3,2.91,3.63,5.67,4.6V98.7c0-2.25,0-3.87,2-5.08V93.51h-6.84v0.12Z" 
transform="translate(-1 -0.61)"/><path d="M151,99.43c0-2.78,2.8-3.67,5-3.67a8.68,8.68,0,0,1,6.1,2.54V93.75a18.86,18.86,0,0,0-5.79-.89c-4.73,0-9.75,1.94-9.75,7,0,8.54,13.36,7.41,13.36,13.7,0,2.82-3.34,4.19-5.56,4.19a13.18,13.18,0,0,1-8.51-3.34l1.13,5a19.46,19.46,0,0,0,7,1.21c6.06,0,10.41-4,10.41-8.34C164.37,104.87,151,104.95,151,99.43Z" 
transform="translate(-1 -0.61)"/><path d="M165.84,93.63c2,0.77,1.71,3.71,1.71,5.56v15.18c0,1.81.24,4.8-1.71,5.52V120h7.38v-0.12c-1.94-.77-1.71-3.71-1.71-5.52V99.19c0-1.81-.24-4.79,1.71-5.56V93.51h-7.38v0.12Z" 
transform="translate(-1 -0.61)"/><path d="M194.16,93.51v0.12c2.06,1.21,2,2.82,2,5.08V114L182,95.36l-0.55-.81a2.6,2.6,0,0,1-.58-1v0h-5.67v0.12c2,1.21,2,2.82,2,5.08v16.07c0,2.25,0,3.86-2.06,5.11V120H182v-0.12c-2.06-1.25-2-2.86-2-5.11V99.19l13.32,17.52c1.71,2.3,2.92,3.63,5.67,4.6V98.7c0-2.25,0-3.87,2-5.08V93.51h-6.84Z" 
transform="translate(-1 -0.61)"/><path d="M12.72,126.16v4.62a2.75,2.75,0,0,1-.34,1.38,2.27,2.27,0,0,1-1,.91,3.4,3.4,0,0,1-1.54.32,2.87,2.87,0,0,1-2.07-.7,2.55,2.55,0,0,1-.74-1.93v-4.6H8.24v4.52a1.81,1.81,0,0,0,.41,1.3,1.69,1.69,0,0,0,1.26.42,1.5,1.5,0,0,0,1.65-1.73v-4.51h1.17Z" 
transform="translate(-1 -0.61)"/><path d="M20.74,133.29H19.31l-3.51-5.69h0l0,0.32q0.07,0.91.07,1.66v3.71H14.79v-7.14h1.42l3.5,5.66h0c0-.08,0-0.35,0-0.82s0-.84,0-1.1v-3.74h1.07v7.14Z" 
transform="translate(-1 -0.61)"/><path d="M23,133.29v-7.14h1.17v7.14H23Z" 
transform="translate(-1 -0.61)"/><path d="M30.42,126.16h1.21l-2.5,7.14H27.9l-2.49-7.14h1.2l1.49,4.44c0.08,0.21.16,0.48,0.25,0.82s0.14,0.58.17,0.75c0-.25.11-0.53,0.2-0.86s0.16-.57.21-0.72Z" 
transform="translate(-1 -0.61)"/><path d="M37,133.29h-4v-7.14h4v1H34.08v1.94h2.69v1H34.08v2.24H37v1Z" 
transform="translate(-1 -0.61)"/><path d="M40,130.44v2.85H38.84v-7.14h2a3.32,3.32,0,0,1,2,.52,1.86,1.86,0,0,1,.66,1.56,2,2,0,0,1-1.39,1.9l2,3.16H42.86l-1.71-2.85H40Zm0-1h0.81a1.85,1.85,0,0,0,1.18-.3,1.1,1.1,0,0,0,.37-0.9,1,1,0,0,0-.4-0.87,2.17,2.17,0,0,0-1.19-.26H40v2.33Z" 
transform="translate(-1 -0.61)"/><path d="M49.68,131.36a1.8,1.8,0,0,1-.69,1.49,3,3,0,0,1-1.9.54,4.53,4.53,0,0,1-2-.38v-1.1a5.34,5.34,0,0,0,1,.36,4.39,4.39,0,0,0,1,.13,1.62,1.62,0,0,0,1-.26,0.86,0.86,0,0,0,.33-0.71,0.89,0.89,0,0,0-.3-0.68A4.57,4.57,0,0,0,47,130.1a3.31,3.31,0,0,1-1.38-.9,1.91,1.91,0,0,1-.4-1.22,1.71,1.71,0,0,1,.63-1.41,2.63,2.63,0,0,1,1.7-.51,5,5,0,0,1,2,.45l-0.37,1a4.47,4.47,0,0,0-1.7-.4,1.31,1.31,0,0,0-.86.25,0.81,0.81,0,0,0-.29.65,0.92,0.92,0,0,0,.12.48,1.2,1.2,0,0,0,.39.37,6.52,6.52,0,0,0,1,.46,5.31,5.31,0,0,1,1.15.61,1.8,1.8,0,0,1,.54.64A1.93,1.93,0,0,1,49.68,131.36Z" 
transform="translate(-1 -0.61)"/><path d="M51.41,133.29v-7.14h1.17v7.14H51.41Z" 
transform="translate(-1 -0.61)"/><path d="M57.26,133.29H56.1v-6.14H54v-1h5.37v1h-2.1v6.14Z" 
transform="translate(-1 -0.61)"/><path d="M62.81,129.41l1.69-3.26h1.27l-2.38,4.37v2.77H62.22v-2.73l-2.37-4.41h1.27Z" 
transform="translate(-1 -0.61)"/><path d="M76.33,129.71a3.9,3.9,0,0,1-.87,2.71,3.66,3.66,0,0,1-5,0,3.93,3.93,0,0,1-.87-2.73,3.86,3.86,0,0,1,.87-2.71A3.21,3.21,0,0,1,73,126a3.14,3.14,0,0,1,2.46,1A3.9,3.9,0,0,1,76.33,129.71Zm-5.45,0a3.19,3.19,0,0,0,.53,2,2.16,2.16,0,0,0,3.15,0,4.05,4.05,0,0,0,0-4A1.86,1.86,0,0,0,73,127a1.9,1.9,0,0,0-1.58.68A3.18,3.18,0,0,0,70.88,129.71Z" 
transform="translate(-1 -0.61)"/><path d="M79.36,133.29H78.2v-7.14h4v1H79.36v2.23h2.69v1H79.36v2.93Z" 
transform="translate(-1 -0.61)"/><path d="M93.39,133.29H92.12L90.91,129c-0.05-.19-0.12-0.45-0.2-0.8s-0.12-.59-0.14-0.73q0,0.31-.16.81c-0.07.33-.13,0.58-0.18,0.74L89,133.29H87.78l-0.92-3.57-0.94-3.56h1.19l1,4.16q0.24,1,.34,1.77c0-.28.09-0.59,0.16-0.93s0.14-.62.2-0.83L90,126.16h1.16l1.19,4.19a15.38,15.38,0,0,1,.36,1.74,12.74,12.74,0,0,1,.35-1.78l1-4.15h1.18Z" 
transform="translate(-1 -0.61)"/><path d="M96.6,133.29v-7.14h1.17v7.14H96.6Z" 
transform="translate(-1 -0.61)"/><path d="M104.11,131.36a1.8,1.8,0,0,1-.69,1.49,3,3,0,0,1-1.9.54,4.53,4.53,0,0,1-2-.38v-1.1a5.4,5.4,0,0,0,1,.36,4.42,4.42,0,0,0,1,.13,1.63,1.63,0,0,0,1-.26,0.86,0.86,0,0,0,.33-0.71,0.89,0.89,0,0,0-.3-0.68,4.59,4.59,0,0,0-1.25-.66,3.29,3.29,0,0,1-1.38-.9,1.91,1.91,0,0,1-.4-1.22,1.71,1.71,0,0,1,.63-1.41,2.63,2.63,0,0,1,1.7-.51,5,5,0,0,1,2,.45l-0.37,1a4.47,4.47,0,0,0-1.7-.4,1.31,1.31,0,0,0-.86.25,0.81,0.81,0,0,0-.29.65,0.92,0.92,0,0,0,.12.48,1.2,1.2,0,0,0,.39.37,6.43,6.43,0,0,0,1,.46,5.31,5.31,0,0,1,1.15.61,1.81,1.81,0,0,1,.54.64A1.93,1.93,0,0,1,104.11,131.36Z" 
transform="translate(-1 -0.61)"/><path d="M108.87,127.05a1.92,1.92,0,0,0-1.58.71,3.75,3.75,0,0,0,0,4,2,2,0,0,0,1.61.67,4.26,4.26,0,0,0,.88-0.09c0.28-.06.58-0.14,0.88-0.23v1a5.34,5.34,0,0,1-1.9.32,3,3,0,0,1-2.41-.95,4,4,0,0,1-.84-2.72,4.4,4.4,0,0,1,.41-1.95,2.92,2.92,0,0,1,1.18-1.28,3.58,3.58,0,0,1,1.81-.44,4.5,4.5,0,0,1,2,.46l-0.42,1a6.37,6.37,0,0,0-.77-0.3A2.75,2.75,0,0,0,108.87,127.05Z" 
transform="translate(-1 -0.61)"/><path d="M118.85,129.71a3.9,3.9,0,0,1-.87,2.71,3.15,3.15,0,0,1-2.47,1,3.18,3.18,0,0,1-2.48-1,3.94,3.94,0,0,1-.87-2.73A3.86,3.86,0,0,1,113,127a3.21,3.21,0,0,1,2.49-.95,3.15,3.15,0,0,1,2.46,1A3.91,3.91,0,0,1,118.85,129.71Zm-5.45,0a3.21,3.21,0,0,0,.53,2,2.16,2.16,0,0,0,3.15,0,4,4,0,0,0,0-4,1.86,1.86,0,0,0-1.56-.68,1.9,1.9,0,0,0-1.59.68A3.18,3.18,0,0,0,113.4,129.71Z" 
transform="translate(-1 -0.61)"/><path d="M126.67,133.29h-1.43l-3.51-5.69h0l0,0.32q0.07,0.91.07,1.66v3.71h-1.06v-7.14h1.42l3.5,5.66h0c0-.08,0-0.35,0-0.82s0-.84,0-1.1v-3.74h1.07v7.14Z" 
transform="translate(-1 -0.61)"/><path d="M133,131.36a1.8,1.8,0,0,1-.69,1.49,3,3,0,0,1-1.9.54,4.52,4.52,0,0,1-2-.38v-1.1a5.31,5.31,0,0,0,1,.36,4.39,4.39,0,0,0,1,.13,1.62,1.62,0,0,0,1-.26,0.86,0.86,0,0,0,.33-0.71,0.88,0.88,0,0,0-.3-0.68,4.53,4.53,0,0,0-1.25-.66,3.31,3.31,0,0,1-1.38-.9,1.92,1.92,0,0,1-.4-1.22,1.71,1.71,0,0,1,.63-1.41,2.64,2.64,0,0,1,1.71-.51,5,5,0,0,1,2,.45l-0.37,1a4.47,4.47,0,0,0-1.7-.4,1.3,1.3,0,0,0-.86.25,0.81,0.81,0,0,0-.29.65,0.92,0.92,0,0,0,.12.48,1.22,1.22,0,0,0,.38.37,6.63,6.63,0,0,0,1,.46,5.26,5.26,0,0,1,1.15.61,1.79,1.79,0,0,1,.54.64A1.9,1.9,0,0,1,133,131.36Z" 
transform="translate(-1 -0.61)"/><path d="M134.73,133.29v-7.14h1.17v7.14h-1.17Z" 
transform="translate(-1 -0.61)"/><path d="M144.07,133.29h-1.43l-3.51-5.69h0l0,0.32c0,0.61.07,1.16,0.07,1.66v3.71h-1.06v-7.14h1.42l3.5,5.66h0c0-.08,0-0.35,0-0.82s0-.84,0-1.1v-3.74h1.07v7.14Z" 
transform="translate(-1 -0.61)"/><path d="M145.75,131.07v-0.93h4.2v0.93h-4.2Z" 
transform="translate(-1 -0.61)"/><path d="M154.72,133.29l-2.07-6h0q0.08,1.33.08,2.49v3.47h-1.06v-7.14h1.64l2,5.68h0l2-5.68H159v7.14h-1.12v-3.53q0-.53,0-1.39c0-.57,0-0.92,0-1h0l-2.14,6h-1Z" 
transform="translate(-1 -0.61)"/><path d="M165.63,133.29l-0.71-2h-2.73l-0.7,2h-1.23l2.67-7.17h1.27l2.67,7.17h-1.24Zm-1-3-0.67-1.94c0-.13-0.12-0.33-0.2-0.62s-0.14-.49-0.18-0.62a11.19,11.19,0,0,1-.38,1.31l-0.64,1.86h2.08Z" 
transform="translate(-1 -0.61)"/><path d="M174,129.66a3.56,3.56,0,0,1-1,2.7,3.94,3.94,0,0,1-2.83.94h-2v-7.14h2.21a3.65,3.65,0,0,1,2.65.92A3.43,3.43,0,0,1,174,129.66Zm-1.23,0q0-2.56-2.4-2.56h-1v5.18h0.83A2.3,2.3,0,0,0,172.73,129.7Z" 
transform="translate(-1 -0.61)"/><path d="M175.83,133.29v-7.14H177v7.14h-1.17Z" 
transform="translate(-1 -0.61)"/><path d="M183.34,131.36a1.8,1.8,0,0,1-.69,1.49,3,3,0,0,1-1.9.54,4.52,4.52,0,0,1-2-.38v-1.1a5.31,5.31,0,0,0,1,.36,4.39,4.39,0,0,0,1,.13,1.62,1.62,0,0,0,1-.26,0.86,0.86,0,0,0,.33-0.71,0.88,0.88,0,0,0-.3-0.68,4.53,4.53,0,0,0-1.25-.66,3.31,3.31,0,0,1-1.38-.9,1.92,1.92,0,0,1-.4-1.22,1.71,1.71,0,0,1,.63-1.41,2.64,2.64,0,0,1,1.71-.51,5,5,0,0,1,2,.45l-0.37,1a4.47,4.47,0,0,0-1.7-.4,1.3,1.3,0,0,0-.86.25,0.81,0.81,0,0,0-.29.65,0.92,0.92,0,0,0,.12.48,1.22,1.22,0,0,0,.38.37,6.63,6.63,0,0,0,1,.46,5.26,5.26,0,0,1,1.15.61,1.79,1.79,0,0,1,.54.64A1.9,1.9,0,0,1,183.34,131.36Z" 
transform="translate(-1 -0.61)"/><path d="M191.4,129.71a3.91,3.91,0,0,1-.87,2.71,3.66,3.66,0,0,1-5,0,3.93,3.93,0,0,1-.87-2.73,3.87,3.87,0,0,1,.87-2.71,3.21,3.21,0,0,1,2.49-.95,3.14,3.14,0,0,1,2.46,1A3.9,3.9,0,0,1,191.4,129.71Zm-5.45,0a3.19,3.19,0,0,0,.53,2,2.16,2.16,0,0,0,3.15,0,4.05,4.05,0,0,0,0-4,1.86,1.86,0,0,0-1.56-.68,1.89,1.89,0,0,0-1.58.68A3.16,3.16,0,0,0,185.95,129.71Z" 
transform="translate(-1 -0.61)"/><path d="M199.22,133.29h-1.43l-3.51-5.69h0l0,0.32q0.07,0.91.07,1.66v3.71h-1.06v-7.14h1.42l3.5,5.66h0c0-.08,0-0.35,0-0.82s0-.84,0-1.1v-3.74h1.07v7.14Z" 
transform="translate(-1 -0.61)"/><path d="M131.41,31.93a49.5,49.5,0,0,0-.86-5.5,39.81,39.81,0,0,0-1.39-4.93,31.28,31.28,0,0,0-2.23-4.93,22.63,22.63,0,0,0-3-4.1,14.94,14.94,0,0,0-11-5.23h-0.09a5.77,5.77,0,0,0-4.16-2.91,4.93,4.93,0,0,0-9.56,0A5.77,5.77,0,0,0,95,7.25H95a14.94,14.94,0,0,0-11,5.23,22.63,22.63,0,0,0-3,4.1,31.28,31.28,0,0,0-2.23,4.93,39.81,39.81,0,0,0-1.39,4.93,49.49,49.49,0,0,0-.86,5.5c-2.3,22.62,7.87,50.42,26.16,54.68A3.17,3.17,0,0,0,104,87.89a3.17,3.17,0,0,0,1.27-1.28C123.54,82.35,133.71,54.55,131.41,31.93ZM104,1.48a3.9,3.9,0,0,1,3.93,2.76,4.86,4.86,0,0,0-3.86,2.47,0.17,0.17,0,0,1-.07.09,0.15,0.15,0,0,1-.07-0.09,4.86,4.86,0,0,0-3.86-2.47
A3.9,3.9,0,0,1,104,1.48Zm-1.86,4.29a3.51,3.51,0,0,1,1.59,2.11,0.29,0.29,0,1,0,.53,0,3.51,3.51,0,0,1,1.59-2.11,4.19,4.19,0,0,1,6,1.58,13.38,13.38,0,0,0-1.67.42,6.6,6.6,0,0,0-2.38,1.32,9.4,9.4,0,0,0-3,6.1c-0.67,7.31,7.72,6.16,8.14,6.13,1.08,0,1.9-1.71,1.9-4s-0.84-4-1.9-4c-0.65,0-1.77.52-1.88,2.55-0.07,1.42.62,3.32,1.52,3.44,0.47,0.06.89-.76,1-1.6s0.06-1.87-.59-2
a0.38,0.38,0,0,0-.46.28,3.83,3.83,0,0,1,.39,1.34
c0,1.25-1.28.63-1.12-1.36,0.15-1.76,1.05-1.65,1.09-1.65,0.5,0,1.26,1,1.26,3,0,1.75-.84,3.63-2.46,2.65-1.36-1-1.89-3.28-1.52-5,0.17-.81.87-3,3.13-3,3.26,0,6.3,1.71,8.72,4.9-0.27.85-1.95,4.1-7.28,7.21l-0.29.15a11,11,0,0,0-4.93-1,27.08,27.08,0,0,0-4.64.74,2.89,2.89,0,0,1-1.84,0,27.08,27.08,0,0,0-4.64-.74,11,11,0,0,0-4.93,1L93.2,24c-5.34-3.11-7-6.36-7.28-7.21,2.42-3.19,5.46-4.9,8.72-4.9,2.26,0,3,2.21,3.13,3,0.38,1.77-.16,4.05-1.52,5-1.61,1-2.46-.9-2.46-2.65,0-2,.76-3,1.26-3,0,0,.94-0.11,1.09,1.65
C96.31,18,95,18.6,95,17.35A3.83,3.83,0,0,1,95.41,16
a0.38,0.38,0,0,0-.46-0.28c-0.65.16-.71,1.3-0.59,2s0.56,1.66,1,1.6c0.9-.12,1.6-2,1.52-3.44-0.1-2-1.23-2.55-1.88-2.55-1.06,0-1.9,1.71-1.9,4
s0.82,4,1.9,4c0.42,0,8.81,1.18,8.14-6.13a9.4,9.4,0,0,0-3-6.1,6.6,6.6,0,0,0-2.38-1.32,13.38,13.38,0,0,0-1.67-.42A4.19,4.19,0,0,1,102.12,5.77ZM86.34,35.9a15.81,15.81,0,0,1-5.8-1.67c0.44-7.31,2.29-13.05,5-16.87,0.48,1.24,2.57,4.35,7.39,7.18C88.82,27,87,32.1,86.34,35.9Zm7.08-10.48a9.35,9.35,0,0,1,4.37-1.21c2.74-.18,4.79.87,6.16,0.91H104c1.37,0,3.4-1.09,6.14-.91
a9.41,9.41,0,0,1,4.39,1.21c5.58,3.56,6.37,11.77,6.48,14.46a43.53,43.53,0,0,1-3.54,19c-3.86,8.51-8.53,14.53-13.14,16.57L104,75.6l-0.37-.16c-4.61-2-9.27-8.06-13.14-16.57a43.52,43.52,0,0,1-3.54-19C87,37.19,87.83,29,93.42,25.42Zm-7.16,11a32.49,32.49,0,0,0-.32,3.31A44.42,44.42,0,0,0,88,55a49.48,49.48,0,0,0,4.13,9.32A11.48,11.48,0,0,1,87,66.64a66.66,66.66,0,0,1-6.47-31.82A16.13,16.13,0,0,0,86.26,36.43Zm6.14,28.35c3.08,5.3,6.12,8.46,8.45,10.14a11.54,11.54,0,0,1-3.54,4.36c-4-2.7-7.4-7-10.07-12.13A11.81,11.81,0,0,0,92.39,64.78Zm8.91,10.46A12.19,12.19,0,0,0,104,76.71a12.2,12.2,0,0,0,2.67-1.47,12,12,0,0,0,3.53,4.34,18.69,18.69,0,0,1-3.58,1.78
s0-.09,0-0.13c-0.26-1.32-2-1.59-2.61-1.59s-2.35.27-2.61,1.59c0,0,0,.09,0,0.13a18.69,18.69,0,0,1-3.58-1.78A12,12,0,0,0,101.31,75.24Zm5.81-.32c2.33-1.67,5.37-4.83,8.45-10.14a11.81,11.81,0,0,0,5.16,2.36c-2.67,5.16-6.06,9.43-10.07,12.13A11.54,11.54,0,0,1,107.12,74.91Zm8.72-10.61A49.48,49.48,0,0,0,120,55a44.42,44.42,0,0,0,2-15.25,32.48,32.48,0,0,0-.32-3.31,16.13,16.13,0,0,0,5.75-1.61A66.66,66.66,0,0,1,121,66.64,11.48,11.48,0,0,1,115.84,64.3Z
m5.78-28.4c-0.62-3.8-2.5-8.8-6.58-11.36,4.82-2.83,6.92-5.94,7.39-7.18,2.69,3.82,4.55,9.56,5,16.87A15.81,15.81,0,0,1,121.62,35.9ZM101,85a23.29,23.29,0,0,1-5.87-2.93,27.5,27.5,0,0,1-3.25-2.62A31.1,31.1,0,0,1,89.53,77q-0.76-.88-1.46-1.81a47.49,47.49,0,0,1-5.58-9.69,63.91,63.91,0,0,1-3.42-10.2,70.46,70.46,0,0,1-1.79-10.85,64.57,64.57,0,0,1-.07-10.84c0.16-1.93.39-3.86,0.74-5.77a39.32,39.32,0,0,1,1.43-5.61,31,31,0,0,1,2.1-4.86,20.93,20.93,0,0,1,3.15-4.44,16.19,16.19,0,0,1,4-3.1A13.93,13.93,0,0,1,90.53,9q0.51-.18,1-0.32
a4.35,4.35,0,0,1,1-.24,14,14,0,0,1,2.56-.23,7.58,7.58,0,0,1,3.88,1,8,8,0,0,1,3.34,6c0.39,4.52-4.21,5.23-5.11,5.22-0.14,0-.21-0.13.24-0.59a6.53,6.53,0,0,0,1-5.1c-0.44-2.07-1.9-3.69-4-3.69A11.16,11.16,0,0,0,86.41,15a22.78,22.78,0,0,0-4.47,7.87,42.69,42.69,0,0,0-2.2,11.38A62.43,62.43,0,0,0,80,44.88a68.71,68.71,0,0,0,1.95,10.59,60.82,60.82,0,0,0,3.53,9.85,43.36,43.36,0,0,0,5.48,9,25.89,25.89,0,0,0,8.13,6.87,18.15,18.15,0,0,0,2.21,1,6.71,6.71,0,0,0,.67,3.1A6.63,6.63,0,0,1,101,85Zm3.29,1.55a0.34,0.34,0,0,1-.62,0,6.49,6.49,0,0,1-1.51-5.17c0.12-.64,1.2-0.93,1.82-0.94s1.7,0.3,1.82.94
A6.49,6.49,0,0,1,104.29,86.55Zm26.38-42.11a70.46,70.46,0,0,1-1.79,10.85,63.9,63.9,0,0,1-3.42,10.2,47.49,47.49,0,0,1-5.58,9.69q-0.7.93-1.46,1.81a31.1,31.1,0,0,1-2.35,2.47,27.5,27.5,0,0,1-3.25,2.62A23.29,23.29,0,0,1,107,85a6.63,6.63,0,0,1-.93.28,6.71,6.71,0,0,0,.67-3.1,18.15,18.15,0,0,0,2.21-1A25.89,25.89,0,0,0,117,74.35a43.36,43.36,0,0,0,5.48-9A60.82,60.82,0,0,0,126,55.47,68.71,68.71,0,0,0,128,44.88a62.43,62.43,0,0,0,.23-10.64
A42.69,42.69,0,0,0,126,22.86,22.78,22.78,0,0,0,121.55,15a11.16,11.16,0,0,0-8.12-3.89c-2.12,0-3.58,1.62-4,3.69
a6.53,6.53,0,0,0,1,5.1c0.45,0.46.38,0.59,0.24,0.59-0.9,0-5.51-.71-5.11-5.22a8,8,0,0,1,3.34-6,7.58,7.58,0,0,1,3.88-1,14,14,0,0,1,2.56.23,4.35,4.35,0,0,1,1,.24q0.52,0.14,1,.32a13.93,13.93,0,0,1,1.93.87,16.19,16.19,0,0,1,4,3.1,20.93,20.93,0,0,1,3.15,4.44,31,31,0,0,1,2.1,4.86A39.33,39.33,0,0,1,130,27.84c0.35,1.91.58,3.84,0.74,5.77A64.57,64.57,0,0,1,130.68,44.45Z" 
transform="translate(-1 -0.61)"/><path d="M112.28,33.43v1.86l0.38-.06h0.18a1.17,1.17,0,0,1,.82.28,1.27,1.27,0,0,1,.21,1.11s-3.74,16.19-4.45,19.27c-0.82-3.9-5.26-25.18-5.26-25.18l0-.09h-0.88v0.1L99.38,55.57,95,36.62a2.7,2.7,0,0,1,0-.28,1.27,1.27,0,0,1,.31-1A1,1,0,0,1,96,35.17l0.37,0v-1.8H90.11v1.76l0.28,0a1.16,1.16,0,0,1,.95.83l7.61,32.67,0,0.09h1.1v-0.1l3.56-23.3,4.53,23.31,0,0.09h1L116.41,36a1,1,0,0,1,1-.75h0.07l0.36,0V33.43h-5.58Z" 
transform="translate(-1 -0.61)"/>
        </symbol>
      </defs>
    </svg>

                '''),
                html.P([
                    'Application feedback, questions or accessibility issues: ',
                    html.A('rolson@waisman.wisc.edu, daifeng.wang@wisc.edu', href='mailto:rolson@waisman.wisc.edu,daifeng.wang@wisc.edu'),
                ]),
                html.P([
                    html.A('Privacy Notice ', href='http://www.wisc.edu/privacy-notice/'),
                    ' | © 2023 Board of Regents of the ',
                    html.A('University of Wisconsin System', href='http://www.wisconsin.edu')
                ])
            ]
        )

    ])

    return layout
