from dash import Dash
from flask_caching import Cache

from application import layout
from application.callbacks import register_callbacks

app = Dash(__name__)
app.title = 'Multimodal Alignment'
server = app.server

CACHE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'cache_dir',
    'CACHE_THRESHOLD': 20  # maximum number of concurrent users of the app
}
cache = Cache()
cache.init_app(server, config=CACHE_CONFIG)


app.layout = layout.get_layout()
register_callbacks(app, cache)


if __name__ == '__main__':
    app.run_server(debug=True)