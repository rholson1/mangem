from dash import Dash
from flask_caching import Cache
import dash_bootstrap_components as dbc
from app_main import layout
from app_main.callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.icons.FONT_AWESOME],
           routes_pathname_prefix='/',
           requests_pathname_prefix='/mangem_aws/')
#           url_base_pathname='/mangem_aws/')
app.title = 'MANGEM'
application = app.server

CACHE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'cache_dir',
    'CACHE_THRESHOLD': 100,  # maximum number of items the cache will store
    'CACHE_DEFAULT_TIMEOUT': 7200  # Default timeout, seconds
}
cache = Cache()
cache.init_app(application, config=CACHE_CONFIG)

app.layout = layout.get_layout()
register_callbacks(app, cache)


if __name__ == '__main__':
    app.run_server(debug=True)
