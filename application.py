from dash import Dash, CeleryManager
from flask_caching import Cache
import dash_bootstrap_components as dbc
from app_main import layout
from app_main.callbacks import register_callbacks
from celery import Celery

REDIS_URL = "redis://localhost:6379"
celery_app = Celery(__name__, broker=REDIS_URL, backend=REDIS_URL)
background_callback_manager = CeleryManager(celery_app)

DEBUG = False
if DEBUG:
    requests_pathname_prefix = None
else:
    requests_pathname_prefix = '/mangem/'

app = Dash(__name__, external_stylesheets=[dbc.icons.FONT_AWESOME], background_callback_manager=background_callback_manager,
           requests_pathname_prefix=requests_pathname_prefix)

app.title = 'MANGEM'
application = app.server


CACHE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'cache_dir',
    'CACHE_THRESHOLD': 500,  # maximum number of items the cache will store
    'CACHE_DEFAULT_TIMEOUT': 172800  # Default timeout, seconds (3600 s/hr * 48 hr = 172800)
}
cache = Cache()
cache.init_app(application, config=CACHE_CONFIG)

app.layout = layout.get_layout()
register_callbacks(app, cache, background_callback_manager)


if __name__ == '__main__':
    app.run_server(debug=True)
