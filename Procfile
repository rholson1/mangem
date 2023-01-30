web: gunicorn --bind :8000 --workers 1 --threads 1 application:application
celery: celery -A application:celery_app worker --loglevel=INFO --concurrency=1