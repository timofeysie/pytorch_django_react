release: python manage.py makemigrations && python manage.py migrate --noinput --ignore=admin
web: gunicorn pytorch_django.wsgi