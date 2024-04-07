### CORS Run through

In my Django project's settings.py file, i have 'corsheaders' to the INSTALLED_APPS list and set the CORS_ALLOWED_ORIGINS like this:

```py
INSTALLED_APPS = [
    'image_classification.apps.ImageClassificationConfig',
    'corsheaders',
    'django.contrib.admin',
    ...
]
SITE_ID = 1
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    ...
]

if 'CLIENT_ORIGIN' in os.environ:
    CORS_ALLOWED_ORIGINS = [
        os.environ.get('CLIENT_ORIGIN')
    ]
else:
    CORS_ALLOWED_ORIGINS = [
        'http://localhost:3000',
    ]
```

The output when the app is run shows:
['http://127.0.0.1:5173']
