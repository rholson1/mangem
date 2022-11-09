from dash import Dash
from flask_caching import Cache
import dash_bootstrap_components as dbc
from application import layout
from application.callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.icons.FONT_AWESOME])
app.title = 'Multimodal Alignment'
server = app.server

CACHE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'cache_dir',
    'CACHE_THRESHOLD': 100,  # maximum number of items the cache will store
    'CACHE_DEFAULT_TIMEOUT': 7200  # Default timeout, seconds
}
cache = Cache()
cache.init_app(server, config=CACHE_CONFIG)


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-YHSJH8FV0K"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
        
          gtag('config', 'G-YHSJH8FV0K');
        </script>

        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = layout.get_layout()
register_callbacks(app, cache)


if __name__ == '__main__':
    app.run_server(debug=True)