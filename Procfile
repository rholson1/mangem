web: gunicorn --bind :8000 --workers 3 application:application
celery: celery -A application:celery_app worker --loglevel=INFO --concurrency=1